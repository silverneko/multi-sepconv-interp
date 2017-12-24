# multi-sepconv-interp
Video frame interpolation using multiple separable convolution

## Data preprocessing
[pyflow](https://github.com/pathak22/pyflow): Fast optical flow estimation

- Youtube videos
- UCF Datasets:
  * [UCF Sports Action Data Set](http://crcv.ucf.edu/data/UCF_Sports_Action.php)
  * [UCF YouTube Action Data Set](http://crcv.ucf.edu/data/UCF_YouTube_Action.php)
- [KITTI Scene Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)

#＃Scripts
### sample-video.py
Sample patch triples from videos.

### list-valid-triple.py
Scan a directory and print a list of triples.

### color-histogram.py
Accept a list of triples (output of list-valid-triple.py) and a threshold value.
Filter triples that potentially contains video shot boundary.
