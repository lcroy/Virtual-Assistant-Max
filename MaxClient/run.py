from Max import Max
from configure import Config

if __name__ == '__main__':
    cfg = Config()
    max = Max(cfg)
    max.call_max(cfg)