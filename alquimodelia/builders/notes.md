# UNET

if using crop(padding) to remove vanishing effect on output crop is really necessary, if you just make the convs converge to the croped image size it will still have the vanishing effect.
But so far it seems the vanishing effect is lost if in the last conv you use padding_style="valid" and change the kernel_size to get that image croping. The thing its that it makes a bigger model, more neurons, it still has a background limit effect, but its hardly noticable. 
More tests on bigger datasets, and on more epochs.
when using the last two convs to provide the croping it has a smaller vanishing efect, but it still does, but it does provides a very good drawing, way better than the crop. should try on scale.