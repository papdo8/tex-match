import cv2 as cv
from glob import glob
import numpy as np
import time
import heapq
from .constants import SUPPORTED_FORMATS
from .argument_parser import parse_args

class Feature:
    def __init__(self,x,y,size):
        self.x = x
        self.y = y
        self.size = size


def load_images(args):
    print('Parsing folder...')
    imgs = [cv.imread(args.input)]
    max_size = imgs[0].shape
    filenames = [args.input]

    # Parse directory for each supported image format
    for filetype in SUPPORTED_FORMATS:
        for filename in glob(args.parse_dir + '\*' + filetype):
            if args.input in filename:
                continue
            else:
                # Read images unchanged for handling alpha channels later
                img = cv.imread(filename, cv.IMREAD_UNCHANGED)
                # Find maximum image size for later upscaling
                if img.shape > max_size:
                    max_size = img.shape

                imgs.append(img)
                filenames.append(filename)

    print('Images loaded.')

    return imgs, filenames, max_size[:2]


def preprocess_images(imgs, max_size):
    print('Preprocessing images...')
    for i, img in enumerate(imgs):
        ''' Some images have an alpha channel that we need to deal with. For the
            image to perceptually stay the same, we have to blend it with a solid
            white background using the equation
            
            new = alpha * original + NOT alpha * background,
            
            where alpha is normalized to [0, 1] range and background is 255 for each
            pixel. The second part can be simplified by scaling the background values
            instead of the alpha values, leaving NOT alpha * 1.
            More on alpha blending: https://en.wikipedia.org/wiki/Alpha_compositing
        '''
        if img.shape[2] == 4:
            # Split color and alpha channels
            img_color = img[:,:,:3]
            alpha = img[:,:,3]
            alpha = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR)
            not_alpha = cv.bitwise_not(alpha)

            # Convert arrays to float for multiplication
            img_color = img_color.astype(float)
            alpha = alpha.astype(float)/255
            not_alpha = not_alpha.astype(float)
            # alpha * original
            img_color = cv.multiply(alpha, img_color)
            # + NOT alpha * 1
            img = cv.add(img_color, not_alpha)

            # Convert back to uint8
            img = img.astype(np.uint8)

        # Convert to grayscale and apply histogram equalization
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.equalizeHist(img)

        # Upscale low resolution images
        if img.shape < max_size:
            img = cv.resize(img, max_size, interpolation=cv.INTER_CUBIC)

        imgs[i] = img

    return imgs


def extract_features(imgs):
    # Very slow.
    print('Extracting SIFT features...')
    start = time.time()
    sift = cv.SIFT_create(nfeatures=200, sigma=6)
    sift_kp = []
    sift_kp_flip = []

    ''' SIFT is not invariant to luminosity, so the descriptors cannot be used here.
        We only need the keypoints and use the knowledge that the more of them fall
        to the same locations, the more likely match an image is to the template.
    '''
    for img in imgs:
        poi = sift.detect(img)
        kp = []
        kp_flip = []
        # Keep only the locations and magnitudes of the keypoints
        for point in poi:
            ft = Feature(point.pt[0], point.pt[1], point.size)
            kp.append(ft)
            # Handle vertical flipping
            ft_flip = Feature(point.pt[0], img.shape[1] - point.pt[1], point.size)
            kp_flip.append(ft_flip)

        sift_kp.append(kp)
        sift_kp_flip.append(kp_flip)

    end = time.time()
    print('Extraction completed in {:.2f} seconds.'.format(end-start))

    return sift_kp[0], sift_kp[1:], sift_kp_flip[1:]


def feature_matching(imgs, filenames, out_dir):
    template_kp, all_kp, all_kp_flip = extract_features(imgs)

    print('Matching features and writing results...')
    start = time.time()

    # Maximum allowed Euclidean distance from the template keypoint
    r = max(imgs[0].shape)*0.005
    # Maximum allowed magnitude difference from the template keypoint
    size_diff_th = 0.2
    scores = []

    for i in range(len(all_kp)):
        counter = 0
        dist_sum = 0
        size_diff_sum = 0

        counter_flip = 0
        dist_sum_flip = 0
        size_diff_sum_flip = 0

        for j in range(len(all_kp[i])):
            for tkp in template_kp:
                # Calculate Euclidean distance from template keypoint
                dist = np.sqrt((tkp.x - all_kp[i][j].x)**2 + (tkp.y - all_kp[i][j].y)**2)
                # Calculate magnitude difference
                size_diff = abs(all_kp[i][j].size - tkp.size)/tkp.size

                # Repeat for flipped keypoints
                dist_flip = np.sqrt((tkp.x - all_kp_flip[i][j].x) ** 2 + (tkp.y - all_kp_flip[i][j].y) ** 2)
                size_diff_flip = abs(all_kp_flip[i][j].size - tkp.size) / tkp.size

                # Discard unrelated keypoints
                if dist < r and size_diff < size_diff_th:
                    counter += 1
                    dist_sum += dist
                    size_diff_sum += size_diff

                if dist_flip < r and size_diff_flip < size_diff_th:
                    counter_flip += 1
                    dist_sum_flip += dist_flip
                    size_diff_sum_flip += size_diff_flip

        if counter > 0:
            # Calculate similarity scores (mean difference)
            dist_score = dist_sum/counter
            size_score = size_diff_sum/counter

            if counter_flip > counter:
                dist_score = dist_sum_flip/counter
                size_score = size_diff_sum_flip/counter

            scores.append([filenames[i+1], counter, dist_score, size_score])

    # Sort scores hierarchically
    sorted_scores = sorted(scores, key=lambda x: (-x[1], x[2], x[3]))
    best_matches = []
    # Select at most 5 best matches
    for i in range(5):
        if i > len(sorted_scores)-1:
            break
        best_matches.append(sorted_scores[i][0])

    end = time.time()
    print('Matching completed in {:.2f} seconds. Writing results...'.format(end - start))

    write_results(filenames[0], best_matches, 'feature matching', out_dir)


def extract_edges(imgs):
    print('Detecting edges...')
    edge_imgs = []

    for img in imgs:
        # Add image size-based Gaussian blur
        ksize = 2 * round(img.shape[0] / 256) + 1
        img = cv.GaussianBlur(img, [ksize, ksize], 2)

        # Calculate Otsu threshold for the Canny edge detector
        th_upper, th_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        th_lower = 0.5 * th_upper

        edges = cv.Canny(img, th_lower, th_upper)
        # Dilate using cross kernel
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        edges = cv.dilate(edges, kernel)

        edge_imgs.append(edges)

    return edge_imgs[0], edge_imgs[1:]


def template_matching(imgs, filenames, out_dir):
    template, edge_imgs = extract_edges(imgs)

    print('Matching template and writing results...')
    start = time.time()

    ''' Due to how cv.matchTemplate() works, we cannot use a template with the same
        size as the images we are inspecting. The template needs to be smaller.
    '''
    cutoff = min(template.shape) // 10
    template = template[cutoff:-cutoff,cutoff:-cutoff]
    # Match the flipped template as well
    template_flip = cv.flip(template, 0)
    scores = {}

    for i, img in enumerate(edge_imgs):
        overlap = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        overlap_flip = cv.matchTemplate(img, template_flip, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(overlap)
        min_val, max_val_flip, min_loc, max_loc = cv.minMaxLoc(overlap_flip)

        # Discard matches with low maxima
        th = 0.05
        if max_val > th:
            scores[filenames[i+1]] = max_val
        if max_val_flip > max_val > th:
            scores[filenames[i+1]] = max_val_flip

    # Select at most 5 best matches
    best_matches = heapq.nlargest(5, scores, key=scores.get)

    end = time.time()
    print('Matching completed in {:.2f} seconds. Writing results...'.format(end - start))

    write_results(filenames[0], best_matches, 'template matching', out_dir)


def write_results(template, best_matches, method, out_dir):
    if not best_matches:
        print('No matches found for the specified template using {} method.'.format(method))
    else:
        # Append unique id (current time) to ensure a new file is generated each time
        out_filename = 'results_{:.0f}.txt'.format(time.time())
        with open(out_dir + '\\' + out_filename, 'w') as f:
            f.write('Matches found (from strongest to weakest) for {} using {} method:'.format(template, method))
            for m in best_matches:
                f.write('\n' + m)

        f.close()
        print('Results written to {}\{}.'.format(out_dir, out_filename))


def main():
    args = parse_args()
    imgs, filenames, max_size = load_images(args)
    imgs = preprocess_images(imgs, max_size)

    if args.f:
        feature_matching(imgs, filenames, args.output_dir)
    else:
        template_matching(imgs, filenames, args.output_dir)


if __name__ == '__main__':
    main()