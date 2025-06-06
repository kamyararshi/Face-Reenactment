from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid_2d, make_coordinate_grid_3d
from modules.util_loss import FaceParser, TernaryLoss, SSIM, smooth_grad_1st
from torchvision import models
import numpy as np
from torch.autograd import grad
import modules.hopenet as hopenet
from modules.gaze import Model
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision import transforms


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        self.downs = self.downs.to(x.device)
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

    @staticmethod
    def make_dict(x, scales):
        out_dict = {}
        assert len(x) == len(scales), f"Scale {len(scales)} and input {len(x)} length should be the same"
        for i in range(len(x)):
            out_dict['prediction_' + str(scales[i]).replace('-', '.')] = x[i]
        return out_dict

class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, device, **kwargs):
        self.device = device
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = (noise + torch.eye(2, 3).view(1, 2, 3)).to(self.device)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']), type=noise.type(), device=self.device)
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
            self.control_params = self.control_params.to(self.device)
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type(), device=frame.device).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection", align_corners=False)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian

# class Transform3D:
#     """
#     Random 3D transformations for equivariance constraints, including frame transformation.
#     """
#     def __init__(self, bs, device, **kwargs):
#         self.device = device
#         # Affine transformation for 3D
#         noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 3, 4]))
#         self.theta = (noise + torch.eye(3, 4).view(1, 3, 4)).to(self.device)
#         self.bs = bs

#         # Thin Plate Spline (TPS) configuration
#         if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
#             self.tps = True
#             self.control_points = make_coordinate_grid_3d(
#                 (kwargs['points_tps'], kwargs['points_tps'], kwargs['points_tps']),
#                 type=noise.type(), device=self.device)
#             self.control_points = self.control_points.unsqueeze(0)
#             self.control_params = torch.normal(
#                 mean=0, std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 3]))
#             self.control_params = self.control_params.to(self.device)
#         else:
#             self.tps = False

#     def transform_frame_3d(self, frame):
#         """
#         Apply a homography derived from the 3D transformation to the 2D frame.
#         """
#         # Derive homography from the 3D affine matrix
#         H = self.theta[:, :2, :3]  # Extract the 2D projection of the 3D matrix
#         grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type(), device=frame.device).unsqueeze(0)
#         grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        
#         # Apply homography to the grid
#         H = H.unsqueeze(1)  # Shape: [bs, 1, 2, 3]
#         transformed_grid = torch.matmul(H[:, :, :, :2], grid.unsqueeze(-1)) + H[:, :, :, 2:]
#         transformed_grid = transformed_grid.squeeze(-1).view(self.bs, frame.shape[2], frame.shape[3], 2)
        
#         return F.grid_sample(frame, transformed_grid, padding_mode="zeros", align_corners=False)


#     def warp_coordinates_2d(self, coordinates):
#         """
#         Warp 2D coordinates for frame transformation.
#         """
#         theta_2d = self.theta[:, :2, :3].type(coordinates.type())  # Use 2D slice of the 3D affine transformation
#         theta_2d = theta_2d.unsqueeze(1)  # Shape: [bs, 1, 2, 3]

#         # Apply affine transformation
#         transformed = torch.matmul(theta_2d[:, :, :, :2], coordinates.unsqueeze(-1)) + theta_2d[:, :, :, 2:]
#         transformed = transformed.squeeze(-1)
#         return transformed

#     def warp_coordinates(self, coordinates):
#         """
#         Warp 3D coordinates using affine and TPS transformations if enabled.
#         """
#         theta = self.theta.type(coordinates.type())
#         theta = theta.unsqueeze(1)  # Shape: [bs, 1, 3, 4]

#         # Apply affine transformation
#         transformed = torch.matmul(theta[:, :, :, :3], coordinates.unsqueeze(-1)) + theta[:, :, :, 3:]
#         transformed = transformed.squeeze(-1)

#         if self.tps:
#             control_points = self.control_points.type(coordinates.type())
#             control_params = self.control_params.type(coordinates.type())
#             distances = coordinates.view(coordinates.shape[0], -1, 1, 3) - control_points.view(1, 1, -1, 3)
#             distances = torch.norm(distances, dim=-1)

#             #NOTE: We have tried 2D RBF kernel with this as well (vox-256_01_12_24_13.37.52_device_1)
#             #NOTE: We have tried 3D RBF Kernel RBF(r)=r (result=distance) and it does not work at all! (vox-256_02_12_24_15.08.58_device_0)
#             result = distances ** 2
#             result = distances * control_params
#             result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
#             transformed = transformed + result

#         return transformed
    

def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=True, add_expression=True):
    kp = kp_canonical['value']    # (bs, k, 3)
    yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
    t, exp = he['t'], he['exp']
    
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    if add_expression: #TODO: For experiments (facial expression improvement)
        exp = exp.view(exp.shape[0], -1, 3)
        kp_transformed = kp_t + exp
    else:
        kp_transformed = kp_t

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']   # (bs, k ,3, 3)
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}

def consistency_loss(motion_coarse, motion_fine):
    # Upsample the coarse motion
    motion_coarse_upsampled = F.interpolate(motion_coarse, size=motion_fine.shape[2:], mode='trilinear', align_corners=True)
    diff = torch.abs(motion_coarse_upsampled - motion_fine)
    return torch.mean(diff)


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_detector, he_estimator, generator, discriminator, train_params, train_stage, estimate_jacobian=True, device=None):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_detector
        self.he_estimator = he_estimator
        self.generator = generator
        self.train_params = train_params
        self.stage = train_stage
        self.scales = train_params['scales']
        if discriminator is not None:
            self.discriminator = discriminator
            self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']['vgg']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available() and device!=None:
                self.vgg = self.vgg.to(device)
                self.vgg.eval()
                
        if self.stage == 'base':
            self.transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            if self.loss_weights['headpose'] != 0:
                self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
                print('Loading hopenet')
                hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
                self.hopenet.load_state_dict(hopenet_state_dict)
                if torch.cuda.is_available() and device!=None:
                    self.hopenet = self.hopenet.to(device)
                    self.hopenet.eval()
        
        elif self.stage=='refiner':
            self._set_train()
            # Loss functions
            weights = self.train_params['gaze_loss']['path'] #"gaze_models/gazeestimation_gazetr.pt"
            if self.train_params['loss_weights']['gaze_loss'] != 0:
                self.gaze = Model(criterion=self.train_params['gaze_loss']['criterion']).to(device)
                self.gaze.load_state_dict(torch.load(weights))
                self.gaze.eval()

            # ID Loss #TODO: Change the face detector to VGGFace or ArcFace (ArcFace had the embedding problem)
            if self.train_params['loss_weights']['id_loss'] != 0:
                # self.face_detector = models.vgg16(pretrained=True).features.to(device).eval()
                # self.face_detector = nn.Sequential(*list(self.face_detector.children())[:30])
                self.face_detector = retinanet_resnet50_fpn(pretrained=True).to(device).eval()

            # Face parser
            self.face_parser = FaceParser(device=device) #NOTE: Check other args

    def _set_train(self):
        """
        Deactivates training for all modules except the generator decoders, up_blocks, and predictors
        """
        for param in self.generator.parameters():
            param.requires_grad = False  # Freeze all generator parameters initially

        # Enable learning only for specific generator submodules
        trainable_submodules = []
        for predictors in self.generator.predict_image:
            for param in predictors.parameters():
                param.requires_grad = True
            predictors.train()
            trainable_submodules.append(predictors)
        for resblock in self.generator.resblock:
            for param in resblock.parameters():
                param.requires_grad = True
            resblock.train()
            trainable_submodules.append(resblock)
        for up_blocks in self.generator.up_blocks:
            for param in up_blocks.parameters():
                param.requires_grad = True
            up_blocks.train()
            trainable_submodules.append(up_blocks)

        # Set evaluation mode for non-trainable modules
        for params in self.kp_extractor.parameters():
            params.requires_grad = False
        for params in self.he_estimator.parameters():
            params.requires_grad = False

        self.kp_extractor.eval()
        self.he_estimator.eval()

        # Ensure generator's other submodules are in eval mode
        for name, module in self.generator.named_children():
            if module not in trainable_submodules:
                module.eval()

    @torch.no_grad()
    def get_embeddings(self, image):
        features = self.face_detector.backbone.body(image)['2']
        embeddings = F.avg_pool2d(features, kernel_size=features.size()[2:]).view(image.size(0), -1)
        return embeddings

    def loss_id(self, i,j, device):
        return F.cosine_embedding_loss(i,j, torch.ones(i.shape[0]).to(device))
    
    def forward(self, x, add_expression=True, rec_driving=False, compute_loss=True):
        kp_canonical = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian, 'heatmap': heatmap}   
        kp_canonical_d = self.kp_extractor(x['driving'])

        he_source = self.he_estimator(x['source'])        # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
        he_driving = self.he_estimator(x['driving'])      # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}

        # {'value': value, 'jacobian': jacobian} #TODO: "add_expression" arg for experiments (facial expression improvement)
        kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian, add_expression=add_expression)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian, add_expression=add_expression)

        generated = self.generator(x['source'], x['driving'], kp_source=kp_source,
                                   kp_driving=kp_driving, rec_driving=rec_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving, 'heatmap_driving': kp_canonical_d['heatmap']})

        if compute_loss and self.stage=='base':
            loss_values = self._compute_loss(
                x=x,
                generated=generated,
                kp_canonical=kp_canonical,
                kp_driving=kp_driving,
                kp_canonical_d=kp_canonical_d,
                he_driving=he_driving,
                add_expression=add_expression
            )
        elif compute_loss and self.stage=='refiner':
            loss_values = self._compute_loss_stage2(
                x=x,
                generated=generated,
            )
        else:
            loss_values = {}

        return loss_values, generated
    
    
    def _compute_loss(self, 
                      x: dict, 
                      generated: dict, 
                      kp_canonical: dict, 
                      kp_driving: dict, 
                      kp_canonical_d: dict, 
                      he_driving: dict,
                      add_expression: bool) -> dict:
        """
        Helper function thath Computes the loss functions during training
        """
        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = ImagePyramide.make_dict(generated['prediction'],
                                                     self.scales[::-1])

        if sum(self.loss_weights['perceptual']['vgg']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual']['vgg'][i] * value
            loss_values['perceptual'] = value_total
        
        # Ternary Loss
        if self.loss_weights['perceptual']['ternary'] != 0:
            value = 0
            for scale in self.scales:
                value += TernaryLoss(
                    pyramide_real['prediction_' + str(scale)].detach(),
                    pyramide_generated['prediction_' + str(scale)],
                ).mean()
            loss_values['perceptual'] += self.loss_weights['perceptual']['l1'] * value / len(self.scales)
        
        if self.loss_weights['perceptual']['l1'] != 0:
            value = 0
            for scale in self.scales:
                value += (pyramide_generated['prediction_' + str(scale)] - pyramide_real['prediction_' + str(scale)].detach()).abs().mean()
            loss_values['perceptual'] += self.loss_weights['perceptual']['l1'] * value / len(self.scales)
            

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if len(generated['deformation'])>1:
            if self.loss_weights['motion_consistency'] != 0:
                num = len(generated['deformation'])-1
                loss_values['motion_consistency'] = \
                    self.loss_weights['motion_consistency'] * \
                sum([
                    consistency_loss(f1, f2) for f1,f2 in zip(generated['deformation'][:-1], generated['deformation'][1:])
                    ])/num
            
        if self.loss_weights['motion_smoothness'] != 0:
            #NOTE: reduce image=True reduces source image resolution to 32x32 then calculates the gradient
            # reduce_image=False reduces the computed gradients (wy, wx) resolution to 31x32 and 32x31
            loss_values['motion_smoothness'] = \
                self.loss_weights['motion_smoothness'] * \
                    smooth_grad_1st(generated['deformation'][-1], x['source'].detach(), reduce_image=False, alpha=10)

        #NOTE: for consistent expression-free Rotation-free canonical keypoints
        if self.loss_weights['canonicalkp_consistency'] != 0:
            # MAE Loss between cacnonical keypoints from the same ID (source and driving)
            loss_values['canonicalkp_consistency'] = self.loss_weights['canonicalkp_consistency'] * torch.abs(kp_canonical['value'] - kp_canonical_d['value']).mean()

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], device=x['source'].device, **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            # transform = Transform3D(x['driving'].shape[0], device=x['source'].device, **self.train_params['transform_params'])
            # transformed_frame = transform.transform_frame_3d(x['driving'])

            transformed_he_driving = self.he_estimator(transformed_frame)

            transformed_kp = keypoint_transformation(kp_canonical, transformed_he_driving, self.estimate_jacobian)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = kp_driving['value'][:, :, :2]
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
                #NOTE: 3D keypoint loss replaced
                # kp_driving_3d = kp_driving['value']  # Full 3D keypoints [batch, num_keypoints, 3]
                # transformed_kp_3d = transformed_kp['value']  # Full 3D keypoints [batch, num_keypoints, 3]

                # # Compute warp directly in 3D
                # warped_kp_3d = transform.warp_coordinates(transformed_kp_3d)
                # value = torch.abs(kp_driving_3d - warped_kp_3d).mean()

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                # project 3d -> 2d
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                    transformed_jacobian_2d)
                
                jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                normed_driving = torch.inverse(jacobian_2d)
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['keypoint'] != 0:
            kp_values = kp_driving['value']
            batch_size, num_kp, _ = kp_values.shape

            # 1.Pairwise distance component
            # Reshape for broadcasting
            kp_a = kp_values.unsqueeze(2)  # Shape: [batch_size, num_kp, 1, 3]
            kp_b = kp_values.unsqueeze(1)  # Shape: [batch_size, 1, num_kp, 3]
            # Calculate squared distances
            squared_dist = torch.sum((kp_a - kp_b) ** 2, dim=-1)  # [batch_size, num_kp, num_kp]
            
            # Apply threshold operation (Dt = 0.1)
            dist_transformed = 0.1 - squared_dist
            mask = torch.gt(dist_transformed, 0)
            # NOTE: Take mean of each pair separately, then sum
            masked_values = dist_transformed * mask
            pair_means = masked_values.mean(dim=0)  # Mean across batch dimension for each pair
            value_total = pair_means.sum()  # Sum all pair means

            # 2.Depth component
            kp_mean_depth = kp_values[:, :, -1].mean(dim=-1)
            value_depth = torch.abs(kp_mean_depth - 0.33).mean()

            # Combine the distance and depth values for the final loss
            value_total = value_total.mean() + value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        if self.loss_weights['headpose'] != 0:
            driving_224 = self.transform_hopenet(x['driving']) # Need the image to be of shape 224x224 due to hopenet

            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)

            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        #TODO: Should use a better way to handle this expression loss. Just minimizing a value not a good idea.
        if self.loss_weights['expression'] != 0 and add_expression:
            value = torch.norm(he_driving['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value

        return loss_values
    
    def _compute_loss_stage2(self,
                             x: dict,
                             generated: dict):
        """
        Computes loss functions for training stage 2
        """
        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = ImagePyramide.make_dict(generated['prediction'],
                                                     self.scales[::-1])

        if sum(self.loss_weights['perceptual']['vgg']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual']['vgg'][i] * value
            loss_values['perceptual'] = value_total
        
        # Ternary Loss
        if self.loss_weights['perceptual']['ternary'] != 0:
            value = 0
            for scale in self.scales:
                value += TernaryLoss(
                    pyramide_real['prediction_' + str(scale)].detach(),
                    pyramide_generated['prediction_' + str(scale)],
                ).mean()
            loss_values['perceptual'] += self.loss_weights['perceptual']['l1'] * value / len(self.scales)
        
        if self.loss_weights['perceptual']['l1'] != 0:
            value = 0
            for scale in self.scales:
                value += (pyramide_generated['prediction_' + str(scale)] - pyramide_real['prediction_' + str(scale)].detach()).abs().mean()
            loss_values['perceptual'] += self.loss_weights['perceptual']['l1'] * value / len(self.scales)
            

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total
                
        if self.loss_weights['gaze_loss'] != 0:
            # Gaze Loss
            gaze_loss = 1 - self.gaze.loss(generated['prediction'][-1], x['driving'].detach())
            loss_values['gaze_loss'] = self.loss_weights['gaze_loss'] * gaze_loss

        if self.loss_weights['wing_loss'] != 0:
            # Wing Loss
            kp_refined_canonical = self.kp_extractor(generated['prediction'][-1])
            he_refined = self.he_estimator(generated['prediction'][-1])
            kp_refined = keypoint_transformation(kp_refined_canonical, he_refined, estimate_jacobian=False)
            
            wing_loss = F.smooth_l1_loss(kp_refined['value'], generated['kp_driving']['value'].detach())
            loss_values['wing_loss'] = self.loss_weights['wing_loss'] * wing_loss
        
        if self.loss_weights['id_loss'] != 0:
            # ID Loss
            embed_gen = self.get_embeddings(generated['prediction'][-1])
            embed_src = self.get_embeddings(x['source'])
            loss_values['id_loss'] = self.loss_weights['id_loss'] *\
                self.loss_id(embed_gen, embed_src, device=generated['prediction'][-1].device)

        if self.loss_weights['pixel_loss'] != 0:
            # Masks only from driving (they should match the reenacted)
            masks = self.face_parser(x['driving'], keep=['eyes', 'lips', 'mouth'])
            refined_masked = generated['prediction'][-1] * masks
            driving_masked = x['driving'] * masks
            pixel_loss = F.smooth_l1_loss(refined_masked, driving_masked.detach(), reduction='sum') / (
                masks.sum() + 1e-8
            )
            loss_values['pixel_loss'] = self.loss_weights['pixel_loss'] * pixel_loss
        

        return loss_values

class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, device):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.device = device
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        self.pyramid = self.pyramid.to(self.device)

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).to(self.device)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid.make_dict(
            [img.detach() for img in generated['prediction'][-len(self.scales):]], 
            self.scales)

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values
