build:
	- docker build -t yanjiai .

run:
	- docker run -it --rm --device=/dev/video0:/dev/video0 yanjiai