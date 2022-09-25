#@markdown ####**Models Settings (note: For pixel art, the best is pixelartdiffusion_expanded):**
diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]

use_secondary_model = True #@param {type: 'boolean'}
diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']
#@markdown #####**Custom model:**
custom_path = '/content/drive/MyDrive/deep_learning/ddpm/ema_0.9999_058000.pt'#@param {type: 'string'}

#@markdown #####**CLIP settings:**
use_checkpoint = True #@param {type: 'boolean'}
ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
ViTL14_336px = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}

#@markdown #####**OpenCLIP settings:**
ViTB32_laion2b_e16 = False #@param{type:"boolean"}
ViTB32_laion400m_e31 = False #@param{type:"boolean"}
ViTB32_laion400m_32 = False #@param{type:"boolean"}
ViTB32quickgelu_laion400m_e31 = False #@param{type:"boolean"}
ViTB32quickgelu_laion400m_e32 = False #@param{type:"boolean"}
ViTB16_laion400m_e31 = False #@param{type:"boolean"}
ViTB16_laion400m_e32 = False #@param{type:"boolean"}
RN50_yffcc15m = False #@param{type:"boolean"}
RN50_cc12m = False #@param{type:"boolean"}
RN50_quickgelu_yfcc15m = False #@param{type:"boolean"}
RN50_quickgelu_cc12m = False #@param{type:"boolean"}
RN101_yfcc15m = False #@param{type:"boolean"}
RN101_quickgelu_yfcc15m = False #@param{type:"boolean"}

#@markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False #@param{type:"boolean"}

diff_model_map = {
    '256x256_diffusion_uncond': { 'downloaded': False, 'sha': 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1', 'uri_list': ['https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'] },
    '512x512_diffusion_uncond_finetune_008100': { 'downloaded': False, 'sha': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648', 'uri_list': ['https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt', 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'] },
    'portrait_generator_v001': { 'downloaded': False, 'sha': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9', 'uri_list': ['https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt'] },
    'pixelartdiffusion_expanded': { 'downloaded': False, 'sha': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a', 'uri_list': ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt'] },
    'pixel_art_diffusion_hard_256': { 'downloaded': False, 'sha': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt'] },
    'pixel_art_diffusion_soft_256': { 'downloaded': False, 'sha': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt'] },
    'pixelartdiffusion4k': { 'downloaded': False, 'sha': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30', 'uri_list': ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt'] },
    'watercolordiffusion_2': { 'downloaded': False, 'sha': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt'] },
    'watercolordiffusion': { 'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt'] },
    'PulpSciFiDiffusion': { 'downloaded': False, 'sha': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f', 'uri_list': ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt'] },
    'secondary': { 'downloaded': False, 'sha': '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a', 'uri_list': ['https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth', 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth'] },
}

kaliyuga_pixel_art_model_names = ['pixelartdiffusion_expanded', 'pixel_art_diffusion_hard_256', 'pixel_art_diffusion_soft_256', 'pixelartdiffusion4k', 'PulpSciFiDiffusion']
kaliyuga_watercolor_model_names = ['watercolordiffusion', 'watercolordiffusion_2']
kaliyuga_pulpscifi_model_names = ['PulpSciFiDiffusion']
diffusion_models_256x256_list = ['256x256_diffusion_uncond'] + kaliyuga_pixel_art_model_names + kaliyuga_watercolor_model_names + kaliyuga_pulpscifi_model_names

from urllib.parse import urlparse

def get_model_filename(diffusion_model_name):
    model_uri = diff_model_map[diffusion_model_name]['uri_list'][0]
    model_filename = os.path.basename(urlparse(model_uri).path)
    return model_filename


def download_model(diffusion_model_name, uri_index=0):
    if diffusion_model_name != 'custom':
        model_filename = get_model_filename(diffusion_model_name)
        model_local_path = os.path.join(model_path, model_filename)
        if os.path.exists(model_local_path) and check_model_SHA:
            print(f'Checking {diffusion_model_name} File')
            with open(model_local_path, "rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == diff_model_map[diffusion_model_name]['sha']:
                print(f'{diffusion_model_name} SHA matches')
                diff_model_map[diffusion_model_name]['downloaded'] = True
            else:
                print(f"{diffusion_model_name} SHA doesn't match. Will redownload it.")
        elif os.path.exists(model_local_path) and not check_model_SHA or diff_model_map[diffusion_model_name]['downloaded']:
            print(f'{diffusion_model_name} already downloaded. If the file is corrupt, enable check_model_SHA.')
            diff_model_map[diffusion_model_name]['downloaded'] = True

        if not diff_model_map[diffusion_model_name]['downloaded']:
            for model_uri in diff_model_map[diffusion_model_name]['uri_list']:
                wget(model_uri, model_path)
                if os.path.exists(model_local_path):
                    diff_model_map[diffusion_model_name]['downloaded'] = True
                    return
                else:
                    print(f'{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri.')
            print(f'{diffusion_model_name} download failed.')


# Download the diffusion model(s)
download_model(diffusion_model)
if use_secondary_model:
    download_model('secondary')

model_config = model_and_diffusion_defaults()
if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': not useCPU,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == '256x256_diffusion_uncond':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': not useCPU,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == 'portrait_generator_v001':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 128,
        'num_heads': 4,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
else:  # E.g. A model finetuned by KaliYuga
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

model_default = model_config['image_size']

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB16: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14_336px: clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x16: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x64: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
if RN101: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB32_laion2b_e16: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion2b_e16').eval().requires_grad_(False).to(device))
if ViTB32_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
if ViTB32_laion400m_32: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
if ViTB32quickgelu_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
if ViTB32quickgelu_laion400m_e32: clip_models.append(open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
if ViTB16_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-16', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
if ViTB16_laion400m_e32: clip_models.append(open_clip.create_model('ViT-B-16', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
if RN50_yffcc15m: clip_models.append(open_clip.create_model('RN50', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
if RN50_cc12m: clip_models.append(open_clip.create_model('RN50', pretrained='cc12m').eval().requires_grad_(False).to(device))
if RN50_quickgelu_yfcc15m: clip_models.append(open_clip.create_model('RN50-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
if RN50_quickgelu_cc12m: clip_models.append(open_clip.create_model('RN50-quickgelu', pretrained='cc12m').eval().requires_grad_(False).to(device))
if RN101_yfcc15m: clip_models.append(open_clip.create_model('RN101', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
if RN101_quickgelu_yfcc15m: clip_models.append(open_clip.create_model('RN101-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)