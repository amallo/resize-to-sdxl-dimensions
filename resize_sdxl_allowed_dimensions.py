from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    InvocationContext,
    ImageField,
    ImageOutput,
    invocation
)

@invocation(
    "image_resize_sdxl",
    title="Image Resize For SDXL",
    tags=["image", "resize", "sdxl"],
    category="image",
    version="1.0.0",
)

class ResizeImageForSDXLInvocation(BaseInvocation):
    """Resizes an image to SDXL allowed dimensions"""
    image: ImageField = InputField(default=None, description="Image to be resize")
   

    def resize_to_allowed_dimensions(width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        image_width, image_height = image.size
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        image_out = image.resize((new_width, new_height))
        image_dto = context.images.save(image=image_out)
        return ImageOutput.build(image_dto)
        
