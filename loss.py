if __name__ == "__main__":
    loss_str = ""
    with open("ResNet50_loss.txt", "r") as f:
        loss_str = f.read()
    loss_list = loss_str.split(",")
    cnt = 0
    res = ""
    for e in loss_list:
        cnt += 1
        if cnt % 64 == 0:
            res += e + ","
    with open("loss.txt", "w") as f:
        f.write(res[:-1])

