class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val):
        self.count += 1
        self.sum += val

    @property
    def avg(self):
        return self.sum / self.count
