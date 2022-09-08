import os
# import sys
# sys.path.append('/home/liyunliang/CGED_Task/pointer-generator-pytorch')
from easy_task.base_module import BaseUtils, TaskSetting
# from grammar_detect_task import GrammarDetectTask
# from seq2action_task import GrammarCorrectTask
# from bart_task import GrammarCorrectTask
from ptr_gen_net_task import GrammarCorrectTask


if __name__ == '__main__':
    # init task utils
    # task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/grammar_ptr_gen_bart.yml'))
    task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/grammar_ptr_predict.yml'))

    # init task setting
    task_setting = TaskSetting(task_utils.task_configuration)

    # build custom task
    # task = GrammarDetectTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train, load_test=True if hasattr(task_setting, 'load_test') and task_setting.load_test else False)
    task = GrammarCorrectTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train, load_test=True if hasattr(task_setting, 'load_test') and task_setting.load_test else False)

    # do train
    if not task_setting.skip_train:
        task.output_result['result_type'] = 'Train_mode'
        task.train(resume_model_path='/home/liyunliang/CGED_Task/output/Grammar_correct_task_ptr-gen_bart-large_5e-5_lang8+linian_cosine_drop0.1_beam3/Model/Grammar_correct_task_ptr-gen_bart-large_5e-5_lang8+linian_cosine_drop0.1_beam3.cpt.dev.0.e(10).b(16).p(1。0).s(99)')
    # do test
    else:
        task.output_result['result_type'] = 'Test_mode'
        task.logger.info('Skip training')
        task.logger.info('Start evaling')

        # load checkpoint and do eval
        task.resume_test_at(task.setting.resume_model_path)