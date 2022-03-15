from .vqa_msd_feature import UniterForVisualQuestionAnswering


class UniterForVisualEntailment(UniterForVisualQuestionAnswering):
    """ Finetune UNITER for VE
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim, 3)
