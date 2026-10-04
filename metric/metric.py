import pyiqa
import piq

## PSNR and LPIPS are computed by pyiqa (https://github.com/chaofengc/IQA-PyTorch). pip install pyiqa
###### data range (0, 1) 
class PSNR():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)

###### data range (0, 1)   
class SSIM():
    def __call__(self, real, fake):
        return piq.ssim(real, fake, data_range=1., reduction='none') 

###### data range (0, 1); pyiqa maps LPIPS inputs to (-1, 1) internally
class LPIPS():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('lpips', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)
