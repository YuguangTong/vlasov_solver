py = python3

clean:
	rm -rf *~
	rm -rf unittest/*~ py_vlasov/*~ py_vlasov/*pyc

test:
	$(py) unittest/test_util.py -v
	$(py) unittest/test_dispersion_tensor.py -v
	$(py) unittest/test_isotropic.py -v
	$(py) unittest/test_parallel.py -v
