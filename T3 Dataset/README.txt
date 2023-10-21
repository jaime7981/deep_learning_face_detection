Simple T3 Dataset

El archivo generate_dataset lee y convierte todas las imagenes a formato jpg,
para luego crear un archivo "data.txt" en formato tsv (tab separated value) con
la ubicación relativa de la imagen y la clase (path, name) estilo train.txt.
Para utilizar, usar el formato mencionado abajo.

El codigo soporta imágenes tipo:

jpg, jpeg, png, blp, bmp, dds, dib, eps, gif,
icns, ico, im, jpeg2000, msp, pcx, apng, ppm, sgi, spider, tga, tiff, webp, xbm,
cur, dcx, fits, fli, flc, fpx, ftex, gbr, gd, imt, iptc, naa, mcidas, mic, mpo, 
pcd, pixar, psd, qoi, sun, wal, wmf, emf, xpm.

Image name format:

	<nombre>_<apellido>_<xx>
	
	* xx es un entero de dos dígitos comenzando en 00
	
Para generar el data.txt simplemente correr el generate dataset:

	python generate_dataset
