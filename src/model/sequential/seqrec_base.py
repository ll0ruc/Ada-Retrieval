
from src.model.base.recommender import BaseRecommender


class SeqRecBase(BaseRecommender):
    def add_annotation(self):
        super(SeqRecBase, self).add_annotation()
        self.annotatins.append('SeqRecBase')
 
