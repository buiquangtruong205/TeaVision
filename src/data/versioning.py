import subprocess


def dvc_pull():
    subprocess.run(["dvc", "pull"], check=True)


def dvc_push():
    subprocess.run(["dvc", "push"], check=True)
