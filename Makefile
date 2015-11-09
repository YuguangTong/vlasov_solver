py = python3

clean:
	rm -rf *~
	rm -rf unittest/*~ py_vlasov/*~ py_vlasov/*pyc

short-test:
	$(py) unittest/test_util.py -v
	$(py) unittest/test_dispersion_tensor.py -v
	$(py) unittest/test_isotropic.py -v
	$(py) unittest/test_parallel.py -v
long-test:
	$(py) unittest/test_follow.py -v 

all-test: short-test long-test
