from models.alphapre import get_model
import torch


model = get_model(
    img_channels=1,
    T_in=6,
    T_out=6,
    input_shape=(384, 384),
)

test_tensor = torch.randn(2, 6, 1, 384, 384)

result = model.predict(frames_in=test_tensor)

print(result[0].shape)
