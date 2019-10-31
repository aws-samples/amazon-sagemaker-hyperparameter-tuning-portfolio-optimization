FROM continuumio/miniconda3:4.6.14

RUN apt-get -y update

RUN conda install pandas==0.24.2 scikit-learn==0.20.0 numpy scipy
RUN pip install boto3==1.9.159

# Set up the program in the image
COPY container/hpo /opt/program
WORKDIR /opt/program

ENTRYPOINT ["python", "-m", "train"]
