#@markdown ####**Custom Model Settings:**
if diffusion_model == 'custom':
  model_config.update({
          'attention_resolutions': '16',
          'class_cond': False,
          'diffusion_steps': 1000,
          'rescale_timesteps': True,
          'timestep_respacing': 'ddim100',
          'image_size': 256,
          'learn_sigma': True,
          'noise_schedule': 'linear',
          'num_channels': 128,
          'num_heads': 1,
          'num_res_blocks': 2,
          'use_checkpoint': use_checkpoint,
          'use_fp16': True,
          'use_scale_shift_norm': False,
      })