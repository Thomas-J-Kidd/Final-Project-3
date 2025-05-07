import os
from ThermalDenoising.schedule.schedule import Schedule
from ThermalDenoising.model.NAFDPM import NAFDPM, EMA
from ThermalDenoising.schedule.diffusionSample import GaussianDiffusion
from ThermalDenoising.schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import copy
from ThermalDenoising.src.sobel import Laplacian
import logging
from collections import OrderedDict
import pyiqa
import wandb

import utils.util as util
from utils.util import crop_concat, crop_concat_back, min_max


def init__result_Dir(path):
    work_dir = os.path.join(path, 'Testing')
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    path = os.path.join(work_dir, str(max_model))
    os.mkdir(path)
    return path


class Tester:
    def __init__(self, config):
        torch.manual_seed(0)
        self.mode = config.MODE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #DEFINE NETWORK
        in_channels = config.CHANNEL_X 
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        
        # Check if we're using UNET or NAFNET
        if hasattr(config, 'MODEL_TYPE') and config.MODEL_TYPE.upper() == 'UNET':
            from ThermalDenoising.model.UnetDPM import UNETDPM
            self.network = UNETDPM(input_channels=in_channels,
                output_channels=out_channels,
                n_channels=config.MODEL_CHANNELS).to(self.device)
        else:
            # Default to NAFNET
            self.network = NAFDPM(input_channels=in_channels,
                output_channels=out_channels,
                n_channels=config.MODEL_CHANNELS,
                middle_blk_num=config.MIDDLE_BLOCKS, 
                enc_blk_nums=config.ENC_BLOCKS, 
                dec_blk_nums=config.DEC_BLOCKS,
                mode=1).to(self.device)
        
        #DEFINE METRICS
        self.psnr = pyiqa.create_metric('psnr', device=self.device)
        self.ssim = pyiqa.create_metric('ssim', device=self.device)
        self.bestPSNR = 0
        
        # Add LPIPS and DISTS metrics if available
        try:
            self.lpips = pyiqa.create_metric('lpips', device=self.device)
            self.dists = pyiqa.create_metric('dists', device=self.device)
            self.has_perceptual_metrics = True
        except:
            self.has_perceptual_metrics = False
            print("LPIPS and/or DISTS metrics not available. Continuing without them.")
        
        #INIT DIFFUSION SAMPLING USING GAUSSIAN DIFFUSION (DDIM)
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.diffusion = GaussianDiffusion(self.network.denoiser, config.TIMESTEPS, self.schedule).to(self.device)

        #LOGGER AND PATHS
        util.setup_logger(
               "base",
                config.LOGGER_PATH,
                "test" + "GoProDeblurring",
                level=logging.INFO,
                screen=True,
                tofile=True,
            )
        self.logger = logging.getLogger("base")
        self.test_img_save_path = config.TEST_IMG_SAVE_PATH
        self.logger_path = config.LOGGER_PATH
        if not os.path.exists(self.test_img_save_path):
            os.makedirs(self.test_img_save_path)
        if not os.path.exists(self.logger_path):
            os.makedirs(self.logger_path)
        self.training_path = config.TRAINING_PATH
        self.pretrained_path_init_predictor = config.PRETRAINED_PATH_INITIAL_PREDICTOR
        self.pretrained_path_denoiser = config.PRETRAINED_PATH_DENOISER
        self.continue_training = config.CONTINUE_TRAINING
        self.continue_training_steps = 0
        self.path_train_gt = config.PATH_GT
        self.path_train_img = config.PATH_IMG
        self.weight_save_path = config.WEIGHT_SAVE_PATH
        self.test_path_img = config.TEST_PATH_IMG
        self.test_path_gt = config.TEST_PATH_GT

        #LR ITERATIONS AND TRAINING STUFFS
        self.iteration_max = config.ITERATION_MAX
        self.LR = config.LR
        self.num_timesteps = config.TIMESTEPS
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.EMA_or_not = config.EMA
        self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH = config.TEST_INITIAL_PREDICTOR_WEIGHT_PATH
        self.TEST_DENOISER_WEIGHT_PATH = config.TEST_DENOISER_WEIGHT_PATH
        self.DPM_SOLVER = config.DPM_SOLVER
        self.DPM_STEP = config.DPM_STEP
        self.beta_loss = config.BETA_LOSS
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        self.native_resolution = config.NATIVE_RESOLUTION
        self.validate_every = config.VALIDATE_EVERY
        
        # GoPro specific settings
        self.grayscale = config.GRAYSCALE if hasattr(config, 'GRAYSCALE') else 'False'
        self.use_gamma = config.USE_GAMMA if hasattr(config, 'USE_GAMMA') else 'False'
 
        #DATASETS AND DATALOADERS
        from ThermalDenoising.data.goprodata import GoProData
        if self.mode == 1:
            dataset_train = GoProData(
                base_path='./dataset/GoPro',
                split='train',
                loadSize=config.IMAGE_SIZE,
                mode=self.mode,
                grayscale=self.grayscale == 'True',
                use_gamma=self.use_gamma == 'True'
            )
            self.batch_size = config.BATCH_SIZE
            self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                               num_workers=config.NUM_WORKERS)
            dataset_test = GoProData(
                base_path='./dataset/GoPro',
                split='test',
                loadSize=config.IMAGE_SIZE,
                mode=0,
                grayscale=self.grayscale == 'True',
                use_gamma=self.use_gamma == 'True'
            )
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        else:
            dataset_test = GoProData(
                base_path='./dataset/GoPro',
                split='test',
                loadSize=config.IMAGE_SIZE,
                mode=self.mode,
                grayscale=self.grayscale == 'True',
                use_gamma=self.use_gamma == 'True'
            )
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        if self.mode == 1 and self.continue_training == 'True':
            print('Continue Training')
            self.network.init_predictor.load_state_dict(torch.load(self.pretrained_path_init_predictor))
            self.network.denoiser.load_state_dict(torch.load(self.pretrained_path_denoiser))
            self.continue_training_steps = config.CONTINUE_TRAINING_STEPS
            
        if self.mode == 1 and config.EMA == 'True':
            self.EMA = EMA(0.9999)
            self.ema_model = copy.deepcopy(self.network).to(self.device)
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        elif config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')
            self.loss = nn.MSELoss()
        if self.high_low_freq == 'True':
            self.high_filter = Laplacian().to(self.device)
        

        
        #WANDB LOGIN AND SET UP
        self.wandb = config.WANDB
        if self.wandb == "True":
            self.wandb = True
            wandb.login()
            run = wandb.init(
                # Set the project where this run will be logged
                project=config.PROJECT,
                # Track hyperparameters and run metadata
                config={
                 "test": 'True',
                 "learning_rate": self.LR,
                 "iterations": self.iteration_max,
                 "Native": self.native_resolution,
                 "DPM_Solver": self.DPM_SOLVER,
                 "Sampling_Steps": config.TIMESTEPS
                })

        else:
            self.wandb = False 

        #DEFINE METRICS
        self.ssim = pyiqa.create_metric('ssim', device=self.device)
        if self.wandb:
            wandb.define_metric("psnr", summary="max")
            wandb.define_metric("ssim", summary="max")
            if self.has_perceptual_metrics:
                wandb.define_metric("lpips", summary="min")
                wandb.define_metric("dists", summary="min")


    def test(self):
        with torch.no_grad():
            #LOAD CHECKPOINTS FOR INITIAL PREDICTOR AND DENOISER
            checkpoint_init = torch.load(self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH)
            checkpoint_denoiser = torch.load(self.TEST_DENOISER_WEIGHT_PATH)
            self.network.init_predictor.load_state_dict(checkpoint_init['model_state_dict'])
            self.network.denoiser.load_state_dict(checkpoint_denoiser['model_state_dict'])
            self.diffusion = GaussianDiffusion(self.network.denoiser, self.num_timesteps, self.schedule).to(self.device)
            print('Test Model loaded')
            
            #PUT EVERYTHING IN EVALUATION MODE
            self.network.eval()
            tq = tqdm(self.dataloader_test)
            sampler = self.diffusion
            iteration = 0
            
            #INIT METRICS DICTIONARY
            test_results = OrderedDict()
            test_results["psnr"] = []
            test_results["ssim"] = []
            if self.has_perceptual_metrics:
                test_results["lpips"] = []
                test_results["dists"] = []
            
            #FOR IMAGES IN TESTING DATASET
            for img, gt, name in tq:
                tq.set_description(f'Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                iteration += 1
                #IF NATIVE DIVIDE IMAGES IN SUBIMAGES
                if self.native_resolution == 'True':
                    temp = img
                    img = crop_concat(img)
                    
                #INIT RANDOM NOISE
                noisyImage = torch.randn_like(img).to(self.device)
                
                #FIRST INITIAL PREDICTION
                init_predict = self.network.init_predictor(img.to(self.device))

                #REFINE RESIDUAL IMAGE USING DPM SOLVER OR DDIM
                if self.DPM_SOLVER == 'True':
                    #DPM SOLVER BRANCH
                    sampledImgs = dpm_solver(self.schedule.get_betas(), self.network.denoiser,
                                             noisyImage, self.DPM_STEP, init_predict, model_kwargs={})
                
                else:
                    #DDIM BRANCH
                    sampledImgs = sampler(noisyImage.cuda(), init_predict, self.pre_ori)
                    
                #COMPUTE FINAL IMAGES   
                finalImgs = (sampledImgs + init_predict)
                
                #IF NATIVE RESOLUTION RECONSTRUCT FINAL IMAGES FROM MULTIPLE SUBIMAGES
                if self.native_resolution == 'True':
                    finalImgs = crop_concat_back(temp, finalImgs)
                    init_predict = crop_concat_back(temp, init_predict)
                    sampledImgs = crop_concat_back(temp, sampledImgs)
                    img = temp

                finalImgs = torch.clamp(finalImgs, 0, 1)
                
                # Save the deblurred image
                save_image(finalImgs, os.path.join(
                    self.test_img_save_path, f"{name[0]}.png"), nrow=1)

                # Save comparison image (blurry input, sharp GT, initial prediction, final deblurred)
                img_compare = torch.cat([img, gt, init_predict.cpu(), finalImgs.cpu()], dim=3)
                save_image(img_compare, os.path.join(
                    self.test_img_save_path, f"{name[0]}_compare.png"), nrow=1)

                # Calculate metrics
                psnr_val = self.psnr(gt.to(self.device), finalImgs.to(self.device)).item()
                ssim_val = self.ssim(gt.to(self.device), finalImgs.to(self.device)).item()
                
                test_results["psnr"].append(psnr_val)
                test_results["ssim"].append(ssim_val)
                
                metric_log = f"img:{name[0]} - PSNR: {psnr_val:.4f} dB; SSIM: {ssim_val:.4f}"
                
                if self.has_perceptual_metrics:
                    lpips_val = self.lpips(gt.to(self.device), finalImgs.to(self.device)).item()
                    dists_val = self.dists(gt.to(self.device), finalImgs.to(self.device)).item()
                    test_results["lpips"].append(lpips_val)
                    test_results["dists"].append(dists_val)
                    metric_log += f"; LPIPS: {lpips_val:.4f}; DISTS: {dists_val:.4f}"
                
                self.logger.info(metric_log)

            # Calculate average metrics
            ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
            ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
            
            self.logger.info(
                "----Average PSNR/SSIM results for {} ----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
                "GoPro deblurring test", ave_psnr, ave_ssim
            ))
            
            if self.has_perceptual_metrics:
                ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
                ave_dists = sum(test_results["dists"]) / len(test_results["dists"])
                self.logger.info("----Average LPIPS\t: {:.6f}\n".format(ave_lpips))
                self.logger.info("----Average DISTS\t: {:.6f}\n".format(ave_dists))
            
            if self.wandb:
                log_dict = {}
                log_dict['psnr'] = ave_psnr
                log_dict['ssim'] = ave_ssim
                if self.has_perceptual_metrics:
                    log_dict['lpips'] = ave_lpips
                    log_dict['dists'] = ave_dists
                wandb.log(log_dict, step=checkpoint_init['iteration'])
                wandb.finish()


def dpm_solver(betas, model, x_T, steps, condition, model_kwargs):
    # Define the noise schedule
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    # Convert the model to the continuous-time noise prediction model
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="x_start",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier-free",
        condition=condition
    )

    # Define dpm-solver and sample by singlestep DPM-Solver
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding")

    # Sample using DPM-Solver
    x_sample = dpm_solver.sample(
        x_T,
        steps=steps,
        order=1,
        skip_type="time_uniform",
        method="singlestep",
    )
    return x_sample
