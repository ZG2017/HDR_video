# HDR_video
# HDR video files:
including:
1. HDR video codes
2. naive HDR video codes
3. HDR images processings
4. example data
5. lookup table
6. test results

# processing pipeline
1. take series of different exposure photos from same camera but with same exposure ratio.
2. fit camera response function using logistic regression based on photos.
3. compute comparametric camera response function(CCRF) lookup table based on camera response function.
4. use CCRF lookup table to generate HDR image from 2 different exposure photos but with same exposure ratio as step 1.
5. apply tune-mapping algorithm to get final output image.
 
