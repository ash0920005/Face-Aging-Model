import torch
import face_aging
import cv2
from torchvision import transforms
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

E = face_aging.Encoder().to(device)
G = face_aging.Generator().to(device)

E.load_state_dict(torch.load("encoder_g.pth"))
G.load_state_dict(torch.load("generator_g.pth"))

E.eval()
G.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])

img = cv2.imread("input.jpg")

img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

img = cv2.resize(img,(64,64))

img = transform(img)

img = img.unsqueeze(0)

img = img.to(device)
target = torch.tensor([0])

target = torch.eye(8)[target]

target = target.float().to(device)
with torch.no_grad():

    z = E(img)

    fake = G(
        z,
        target
    )
    output = fake.squeeze(0)

output = (output + 1) / 2

save_image(
    output,
    "aged.png"
)