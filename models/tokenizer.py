from torch import nn
from torch.nn import functional as F

class Tokenizer(nn.Module):
    def __init__(self, tokenizer_name, tokenizer_ckpt_path=""):
        super(Tokenizer, self).__init__()
        if tokenizer_name.startswith("llamagen"):
            from models.llamagen.tokenizer import LlamaGenTokenizer
            self.model = LlamaGenTokenizer(tokenizer_name, tokenizer_ckpt_path)
        elif tokenizer_name.startswith("gigatok"):
            from models.gigatok.tokenizer import GigaTokTokenizer
            self.model = GigaTokTokenizer(tokenizer_name, tokenizer_ckpt_path)
        elif tokenizer_name.startswith("titok"):
            from models.titok.tokenizer import TiTokTokenizer
            self.model = TiTokTokenizer(tokenizer_name, tokenizer_ckpt_path)
        elif tokenizer_name.startswith("var"):
            from models.var.tokenizer import VARTokenizer
            self.model = VARTokenizer(tokenizer_name, tokenizer_ckpt_path)
        elif tokenizer_name.startswith("imf"):
            from models.imf.tokenizer import IMFTokenizer
            self.model = IMFTokenizer(tokenizer_name, tokenizer_ckpt_path)
        else:
            raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")

    def train_mode(self):
        return self.model.train_mode()

    def eval_mode(self):
        return self.model.eval_mode()

    def trainable_parameters(self):
        return self.model.trainable_parameters()

    @property
    def last_layer(self):
        return self.model.last_layer

    def standard_reconstruction(self, x):
        return self.model.standard_reconstruction(x)

    def extract_latents(self, x):
        return self.model.extract_latents(x)

    def latent_to_image(self, latents):
        return self.model.latent_to_image(latents)

    def forward(self, x):
        return self.model.standard_reconstruction(x)

    def collect_eval_info(self, x):
        x_rec = self.model.standard_reconstruction(x)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
