# These are default settings, all the settings can be changed via Main

CONSTRAIN_MODE = None  # single/hybrid/None
CONSTRAIN_TYPE = None  # fit/err

# Number of neighbors for KNN
NUM_NEIGHBORS = 3

# Local search parameters
LOCAL_SEARCH = False
LOCAL_TYPE = 'asym' # can be asym - asymetric or std - standard
LOCAL_ASYM_FLIP = None
LOCAL_STUCK_THRESHOLD = 5
LOCAL_ITERATIONS = 1000
TOP_POP_RATE = 0.1

# Surrogate model
SURROGATE_UPDATE_DURATION = 40
SURROGATE_VERSION = 'std' # std - standard (fea1-fea2), sel - consider selected features

# Length change threshold
LENGTH_UPDATE = False
LENGTH_STUCK_THRESHOLD = 10
LENGTH_ITERATIONS = 100
