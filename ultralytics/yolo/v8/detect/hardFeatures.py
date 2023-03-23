
import cv2
from skimage.metrics import structural_similarity




class SIFTFeatures:
    def getFeatures(self, img):
        print()
          # Initialize SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)

        return keypoints, descriptors

    def structural_sim(self, img1, img2):

        sim, diff = structural_similarity(img1, img2, full=True)
        print("die Ähnlichkeit ist" + str(sim))
        return sim

    
    def matching(self, feature1, feature2):
        
        
        # Extract features from both images
        keypoints1, descriptors1 = feature1
        keypoints2, descriptors2 = feature2

        # Initialize Flann based matcher
        flann = cv2.FlannBasedMatcher()

        # Match descriptors
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Ratio test for good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Return true if enough good matches
        if len(good_matches) >= 10:
            return True
        else:
            return False


