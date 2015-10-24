clean:
	rm *~
	rm -rf unittest/*~ py_vlasov/*~ py_vlasov/*pyc

test:
	python3 unittest/test_util.py
	python3 unittest/test_dispersion_tensor.py
