from pix2pix_model import Pix2PixModel
from dataset_utils import load_rgba_ds
import timeit
import statistics

train_ds, test_ds = load_rgba_ds(2, 3, should_augment_hue=False, should_augment_translation=False)

model = Pix2PixModel(train_ds, test_ds, "baseline", "front-to-right", 100., False)
model.load_generator()

num_runs = 10

unbatched = train_ds.unbatch()
batch1 = iter(unbatched.batch(1).take(num_runs + 1))
batch4 = iter(unbatched.batch(4).take(num_runs + 1))


def run_1():
    inp, _ = next(batch1)
    model.generator(inp, training=True)


def run_4():
    inp, _ = next(batch4)
    model.generator(inp, training=True)


run_1()
run_4()


def best(t, digits=4):
    return round(min(t), digits)


def avg(t, digits=4):
    return round(sum(t) / len(t), digits)


def std(t, digits=4):
    return round(statistics.pstdev(t), digits)


t1 = timeit.repeat(run_1, repeat=num_runs, number=1)

print("best batch1", best(t1))
print("avg batch1", avg(t1))
print("std batch1", std(t1))

t4 = timeit.repeat(run_4, repeat=num_runs, number=1)
# t4 = timeit.repeat(run_4)
#
#


# print("t1", t1)
#
print("best batch4", best(t4))
print("avg batch4", avg(t4))
print("std batch4", std(t4))
