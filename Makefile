clean:
	rm -rf *~
	rm -rf unittest/*~ py_vlasov/*~ py_vlasov/*pyc

test:
	python3 unittest/test_util.py -v
	python3 unittest/test_dispersion_tensor.py -v
	python3 unittest/test_isotropic.py -v
