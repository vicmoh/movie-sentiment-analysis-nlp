run:
	python3 ./src/main.py

split_data:
	python3 ./src/split_data.py

stackbuse_example:
	python3 ./src/example/stackbuse.py

sen_example:
	python3 ./src/example/sen_example.py

scikit_example:
	python3 ./src/example/scikit_example.py

git:
	git add -A
	git commit -m "[AUTO]"
	git push

push:
	git add -A
	git commit -m "$(m)"
	git push

dependencies:
	pip3 install numpy
	pip3 install scipy
	pip3 install -U scikit-learn
	pip3 install matplotlib 
	pip3 install pandas
	pip3 install virtualenv
	pip3 install --user -U nltk
