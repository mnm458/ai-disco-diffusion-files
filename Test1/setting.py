#@markdown ####**Basic Settings:**
batch_name = 'TimeToDisco' #@param{type: 'string'}
steps = 250 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
width_height_for_512x512_models = [1280, 768] #@param{type: 'raw'}
clip_guidance_scale = 5000 #@param{type: 'number'}
tv_scale = 0#@param{type: 'number'}
range_scale = 150#@param{type: 'number'}
sat_scale = 0#@param{type: 'number'}
cutn_batches = 4#@param{type: 'number'}
skip_augs = False#@param{type: 'boolean'}

#@markdown ####**Image dimensions to be used for 256x256 models (e.g. pixelart models):**
width_height_for_256x256_models = [512, 448] #@param{type: 'raw'}

#@markdown ####**Video Init Basic Settings:**
video_init_steps = 100 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
video_init_clip_guidance_scale = 1000 #@param{type: 'number'}
video_init_tv_scale = 0.1#@param{type: 'number'}
video_init_range_scale = 150#@param{type: 'number'}
video_init_sat_scale = 300#@param{type: 'number'}
video_init_cutn_batches = 4#@param{type: 'number'}
video_init_skip_steps = 50 #@param{type: 'integer'}

#@markdown ---

#@markdown ####**Init Image Settings:**
init_image = None #@param{type: 'string'}
init_scale = 1000 #@param{type: 'integer'}
skip_steps = 10 #@param{type: 'integer'}
#@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*

width_height = width_height_for_256x256_models if diffusion_model in diffusion_models_256x256_list else width_height_for_512x512_models

#Get corrected sizes
side_x = (width_height[0]//64)*64;
side_y = (width_height[1]//64)*64;
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

#Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)