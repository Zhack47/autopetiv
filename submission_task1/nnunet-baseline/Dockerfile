FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /opt/algorithm /input /output /output/images /output/images/tumor-lesion-segmentation
RUN chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt && \
    mkdir -p /opt/algorithm/nnUNet_raw && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result


COPY --chown=algorithm:algorithm nnunetv2-2.6.2 /opt/algorithm/nnunetv2-2.6.2
RUN python -m pip install -e nnunetv2-2.6.2

COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/

ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

ENTRYPOINT ["python", "-m", "process"]
