"""
Codes to split OULU-NPU dataset by 4 protocols
Reference: https://github.com/clks-wzz/FAS-SGTD/blob/master/fas_sgtd_single_frame/util/util_dataset.py
"""
import os

protocol_dict = {}
protocol_dict['protocol_1']={'train': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6], 
                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                            'dev': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6], 
                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                            'test': { 'session': [3], 'phones': [1, 2, 3, 4, 5, 6], 
                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                            }
protocol_dict['protocol_2']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                    'users': list(range(1,21)), 'PAI': [1, 2, 4] },
                            'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                    'users': list(range(21,36)), 'PAI': [1, 2, 4] },
                            'test': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                    'users': list(range(36,56)), 'PAI': [1, 3, 5] }
                            }        
protocol_dict['protocol_3']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5], 
                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                            'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5], 
                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                            'test': { 'session': [1, 2, 3], 'phones': [6], 
                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                            }
protocol_dict['protocol_4']={'train': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5], 
                                    'users': list(range(1,21)), 'PAI': [1, 2, 4] },
                            'dev': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5], 
                                    'users': list(range(21,36)), 'PAI': [1, 2, 4] },
                            'test': { 'session': [3], 'phones': [6], 
                                    'users': list(range(36,56)), 'PAI': [1, 3, 5] }
                            }


class OULU_Split:

    def __init__(self):
     







if __name__ == "__main__":

    print(protocol_dict)