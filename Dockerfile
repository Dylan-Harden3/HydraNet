FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip python3-venv

WORKDIR /workspace

COPY . /workspace/

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install black tqdm

CMD ["bash"]