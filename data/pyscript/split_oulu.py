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
    def __init__(self, train_file_path, dev_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
    
    def use_protocol1(self, save_train_path, dev_save_path, test_save_path):
        with open(train_file_path, 'r') as fr, open(save_train_path, 'w') as fw:
            lines = fr.readlines()
            for line in lines:
                atts = [int(x) for x in line.split("/")[1].split("_")]
                atts_phone = atts[0]
                atts_session = atts[1]
                atts_user = atts[2]
                atts_pai = atts[3]
                if atts_phone in protocol_dict['protocol_1']['train']['phones'] and atts_session in protocol_dict['protocol_1']['train']['session'] and \
                    atts_user in protocol_dict['protocol_1']['train']['users'] and atts_pai in protocol_dict['protocol_1']['train']['PAI']:
                    fw.writelines(line)

        with open(dev_file_path, 'r') as fr, open(save_dev_path, 'w') as fw:
            lines = fr.readlines()
            for line in lines:
                atts = [int(x) for x in line.split("/")[1].split("_")]
                atts_phone = atts[0]
                atts_session = atts[1]
                atts_user = atts[2]
                atts_pai = atts[3]
                if atts_phone in protocol_dict['protocol_1']['dev']['phones'] and atts_session in protocol_dict['protocol_1']['dev']['session'] and \
                    atts_user in protocol_dict['protocol_1']['dev']['users'] and atts_pai in protocol_dict['protocol_1']['dev']['PAI']:
                    fw.writelines(line)

        with open(test_file_path, 'r') as fr, open(save_test_path, 'w') as fw:
            lines = fr.readlines()
            for line in lines:
                atts = [int(x) for x in line.split("/")[1].split("_")]
                atts_phone = atts[0]
                atts_session = atts[1]
                atts_user = atts[2]
                atts_pai = atts[3]
                if atts_phone in protocol_dict['protocol_1']['test']['phones'] and atts_session in protocol_dict['protocol_1']['test']['session'] and \
                    atts_user in protocol_dict['protocol_1']['test']['users'] and atts_pai in protocol_dict['protocol_1']['test']['PAI']:
                    fw.writelines(line)


if __name__ == "__main__":

    train_file_path = "/home/projects/face_liveness/FAS/data/list_oulu/train_list.txt"
    dev_file_path = "/home/projects/face_liveness/FAS/data/list_oulu/dev_list.txt"
    test_file_path = "/home/projects/face_liveness/FAS/data/list_oulu/test_list.txt"
    oulu = OULU_Split(train_file_path, dev_file_path, test_file_path)

    save_train_path = "/home/projects/face_liveness/FAS/data/list_oulu/p1_train_list.txt"
    save_dev_path = "/home/projects/face_liveness/FAS/data/list_oulu/p1_dev_list.txt"
    save_test_path = "/home/projects/face_liveness/FAS/data/list_oulu/p1_test_list.txt"
    oulu.use_protocol1(save_train_path, save_dev_path, save_test_path)