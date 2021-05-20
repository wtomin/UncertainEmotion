import collections

Dataset_Info = collections.namedtuple("Dataset_Info", 'data_file categories')
class PATH(object):
    def __init__(self):
        self.Aff_wild2 = Dataset_Info(data_file = 'annotations/annotations.pkl',
            categories = {'AU': ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']})


