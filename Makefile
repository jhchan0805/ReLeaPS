SHELL=/bin/bash -o pipefail
PYTHON:=PYTHONUNBUFFERED=1 /usr/bin/env python3

.DELETE_ON_ERROR:

.PHONY: all train benchmark clean

all: train benchmark

train: data/models/blobs_20.bin data/models/sculpture_20.bin

benchmark: \
	data/benchmark/blobs_all.txt \
	data/benchmark/blobs_rand_20.txt \
	data/benchmark/blobs_rl_20_test.txt \
	data/benchmark/sculpture_all.txt \
	data/benchmark/sculpture_rand_20.txt \
	data/benchmark/sculpture_rl_20_test.txt

clean:
	rm -rf data/*_processed/ data/*render*/ data/*_q_vals_*/ data/models/* data/benchmark/*.txt

data/diligent_test/000000.npz: src/diligent_process.py data/DiLiGenT.zip
	rm -rf data/diligent_test/
	mkdir -p data/diligent_test/
	$(PYTHON) -m src.diligent_process data/DiLiGenT.zip DiLiGenT/pmsData/ data/diligent_test/

data/benchmark/%_all.txt: src/benchmark/__main__.py data/%_test/000000.npz
	$(PYTHON) -m src.benchmark all data/$*_test/ | tee data/benchmark/$*_all.txt

data/benchmark/%_rand_20.txt: src/benchmark/__main__.py data/%_test/000000.npz
	$(PYTHON) -m src.benchmark random data/$*_test/ 20 100 | tee data/benchmark/$*_rand_20.txt

data/models/20/%.bin: src/benchmark/__main__.py src/rl_train.py src/rl_model.py src/rl_env.py \
		data/blobs_train/000000.npz data/sculpture_train/000000.npz
	$(PYTHON) -m src.rl_train data/blobs_train/+data/sculpture_train/ 20 $* data/models/20/$*.bin | \
		tee data/benchmark/rl_20_train_$*.txt

data/benchmark/%_rl_20_test.txt: src/benchmark/__main__.py src/rl_test.py src/rl_model.py src/rl_env.py \
		data/%_test/000000.npz
	rm -rf data/$*_q_vals_20/
	mkdir -p data/$*_q_vals_20/
	$(PYTHON) -m src.rl_test data/$*_test/ 20 data/models/20/ data/$*_q_vals_20/ | \
		tee data/benchmark/$*_rl_20_test.txt

