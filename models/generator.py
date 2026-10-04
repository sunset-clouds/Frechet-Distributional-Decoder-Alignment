from torch import nn

class Generator(nn.Module):
    def __init__(self, generator_name, generator_ckpt_path="", resolution=256, cfg_omega=None):
        super(Generator, self).__init__()
        if generator_name.startswith("llamagen"):
            from models.llamagen.generator import LlamaGenGenerator
            self.model = LlamaGenGenerator(generator_name, generator_ckpt_path, resolution,
                                           cfg_omega)
        elif generator_name.startswith("gigatok"):
            from models.gigatok.generator import GigaTokGenerator
            self.model = GigaTokGenerator(generator_name, generator_ckpt_path, resolution,
                                          cfg_omega)
        elif generator_name.startswith("titok"):
            from models.titok.generator import TiTokGenerator
            self.model = TiTokGenerator(generator_name, generator_ckpt_path, resolution,
                                        cfg_omega)
        elif generator_name.startswith("var"):
            from models.var.generator import VARGenerator
            self.model = VARGenerator(generator_name, generator_ckpt_path, resolution, cfg_omega)
        elif generator_name.startswith("imf"):
            from models.imf.generator import IMFGenerator
            self.model = IMFGenerator(generator_name, generator_ckpt_path, resolution, cfg_omega)
        else:
            raise ValueError(f"Unsupported generator_name: {generator_name}")

    def eval_mode(self):
        return self.model.eval_mode()

    def standard_generation(self, batch_size, labels=None, use_cfg=True, device=None, seeds=None):
        return self.model.standard_generation(
            batch_size=batch_size,
            labels=labels,
            use_cfg=use_cfg,
            device=device,
            seeds=seeds,
        )
