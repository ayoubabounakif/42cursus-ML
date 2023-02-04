from math import sqrt

class TinyStatistician:

    def __init__(self):
        pass

    # µ = Σx / m
    def mean(self, x):
        if not len(x) != 0: return None
        return sum(x) / len(x)

    def median(self, x):
        if not len(x) != 0: return None
        x = [float(i) for i in x]
        x = sorted(x)
        if len(x) % 2 == 0:
            return (x[int(len(x) / 2)] + x[int(len(x) / 2) - 1]) / 2
        else:
            return x[int(len(x) / 2)]

    # Q1 = Σx / 4
    # Q3 = Σx / 4 * 3
    def quartiles(self, x):
        if not len(x) != 0: return None
        x = [float(i) for i in x]
        x = sorted(x)
        if len(x) % 2 == 0:
            q1 = (x[int(len(x) / 4)] + x[int(len(x) / 4) - 1]) / 2
            q3 = (x[int(len(x) / 4 * 3)] + x[int(len(x) / 4 * 3) - 1]) / 2
        else:
            q1 = x[int(len(x) / 4)]
            q3 = x[int(len(x) / 4 * 3)]
        return [q1, q3]

    # Lp_th = (n + 1) * p_th / 100
    # Pp_th = X[int(Lp_th)] + (Lp_th - int(Lp_th)) * (X[int(Lp_th) + 1] - X[int(Lp_th)])
    def percentile(self, x, p):
        if not len(x) != 0: return None
        x = [float(i) for i in x]
        x = sorted(x)
        Lp = (len(x) - 1) * p / 100
        index = int(Lp)
        Pp = x[index] + (Lp - index) * (x[index + 1] - x[index])
        return round(Pp, 1)

    # σ² = Σ(x - µ)² / m - 1
    def var(self, x):
        if not len(x) != 0: return None
        x = [float(i) for i in x]
        return sum([(i - self.mean(x)) ** 2 for i in x]) / (len(x) - 1)

    # σ = √σ²
    def std(self, x):
        if not len(x) != 0: return None
        x = [float(i) for i in x]
        return sqrt(self.var(x))



if __name__ == '__main__':
    tstat = TinyStatistician()
    dataset = [1, 42, 300, 10, 59]

    print('------Data-----', dataset)
    print(tstat.mean(dataset)) # 82.4
    print(tstat.median(dataset)) # 42.0
    print(tstat.quartiles(dataset)) # [10.0, 59.0]
    print(tstat.var(dataset)) # 12279.439999999999
    print(tstat.std(dataset)) # 110.81263465868862
    print(tstat.percentile(dataset, 10)) # 4.6
    print(tstat.percentile(dataset, 15)) # 6.4
    print(tstat.percentile(dataset, 20)) # 8.2

    dataset = []
    print('------Data-----', dataset)
    print(tstat.mean(dataset)) # None
    print(tstat.median(dataset)) # None
    print(tstat.quartiles(dataset)) # None
    print(tstat.var(dataset)) # None
    print(tstat.std(dataset)) # None
    print(tstat.percentile(dataset, 10)) # None
    print(tstat.percentile(dataset, 15)) # None
    print(tstat.percentile(dataset, 20)) # None