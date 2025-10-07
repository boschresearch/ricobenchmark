#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0



import logging
import os, sys

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import file_io
from detectron2.utils import comm

# from detectron2.continual_learning import EWC

from detectron2.modeling import GeneralizedRCNNWithTTA, ema

from detectron2.continual_learning import inference_on_continual_learning_dataset


import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def do_test(cfg, model, eval_only=False, mode='val', only_id=None, json_name='', global_ignore_class_ids=None, eval_single_class=False, task_id=None):
    logger = logging.getLogger("detectron2")
    
    dataloader = cfg.dataloader.eval if mode=='val' else cfg.dataloader.test

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        
        ret = inference_on_continual_learning_dataset(
            cfg, model, instantiate(dataloader), only_id, json_name, mode, global_ignore_class_ids, eval_single_class=eval_single_class, task_id=task_id
        )
        for key, value in ret.items():
            print(key)
            print_csv_format(value)

        return ret

    logger.info("Run evaluation without EMA.")
    ret = inference_on_continual_learning_dataset(
        cfg, model, instantiate(dataloader), only_id, json_name, mode, task_id=task_id)
    for key, value in ret.items():
        print(key)
        print_csv_format(value)
        
    if cfg.train.model_ema.enabled:
        logger.info("Run evaluation with EMA.")
        with ema.apply_model_ema_and_restore(model):
            if "evaluator" in cfg.dataloader:
                ema_ret = inference_on_continual_learning_dataset(
                    cfg, model, instantiate(dataloader), only_id, json_name, mode)
                for key, value in ema_ret.items():
                    print(key)
                    print_csv_format(value)
                ret.update(ema_ret)
    return ret


def print_model_parameter_info(model, logger):
    logger.info("Model has {} parameters".format(sum(p.numel() for p in model.parameters())))
    logger.info("Backbone has {} parameters".format(sum(p.numel() for p in model.backbone.parameters())))
    logger.info("RPN has {} parameters".format(sum(p.numel() for p in model.proposal_generator.parameters())))
    logger.info("ROI-Heads has {} parameters".format(sum(p.numel() for p in model.roi_heads.parameters())))
    
    logger.info("Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.info("Backbone has {} trainable parameters".format(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)))
    logger.info("RPN has {} trainable parameters".format(sum(p.numel() for p in model.proposal_generator.parameters() if p.requires_grad)))
    logger.info("ROI-Heads has {} trainable parameters".format(sum(p.numel() for p in model.roi_heads.parameters() if p.requires_grad)))  

    
def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")

    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    print_model_parameter_info(model, logger)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    # build model ema
    ema.may_build_model_ema(cfg, model)
    
    cfg.train.trainer.model = model
    cfg.train.trainer.data_loader = train_loader
    cfg.train.trainer.optimizer = optim
    cfg.train.trainer.cfg = cfg

    trainer = instantiate(cfg.train.trainer)
        
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )
    trainer.register_hooks(
        [
            hooks.LoadCheckpoint(checkpointer) if cfg.train.forward_transfer else None,
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.CLLRScheduler(optimizer=cfg.optimizer, scheduler=cfg.lr_multiplier),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHookContinualLearning(cfg.train.eval_period, lambda only_id=None, task_id=None, json_name='': do_test(cfg, model, only_id=only_id, json_name=json_name, task_id=task_id)) if cfg.train.eval_period <= cfg.train.max_iter else None,
            hooks.CLBestCheckpointer(eval_period=cfg.train.eval_period, checkpointer=checkpointer, val_metric="bbox/AP", mode="max") if cfg.train.save_and_load_best_model else None,
            hooks.LoadBestModel(checkpointer, cfg.train.output_dir) if cfg.train.save_and_load_best_model else None,
            hooks.TestHookContinualLearning(lambda only_id=None, task_id=None, json_name='': do_test(cfg, model, only_id=only_id, mode='test', json_name=json_name, task_id=task_id)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter,
                                use_wandb=args.wandb, do_cl=True),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            
            # hooks.LoadSaveBackboneRPNHead(period=cfg.train.log_period, save_dir=cfg.train.output_dir,
            #                           backbone_path=cfg.train.backbone_path, rpn_path=cfg.train.rpn_path,
            #                           roi_heads_path=cfg.train.roi_heads_path),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
        

    if cfg.train.reinit_head:    
        reinit_weights(model.roi_heads)

    if cfg.train.reinit_rpn:
        reinit_weights(model.proposal_generator)

    if cfg.train.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Freezing backbone")
        print_model_parameter_info(model, logger)

    if cfg.train.only_train_outputs:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.proposal_generator.parameters():
            param.requires_grad = False
        for param in model.roi_heads.parameters():
            param.requires_grad = False
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True

        logger.info("Only training box predictor")
        print_model_parameter_info(model, logger)
        
    
    trainer.train(start_iter, cfg.train.max_iter)
    
    
def reinit_weights(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    else:
        for layer in module.children():
            reinit_weights(layer)

    
def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.train.output_dir = file_io.experiment_folder(cfg.train.exp_folder, cfg.train.output_dir)
    default_setup(cfg, args)

    if cfg.train.eval_only:
        args.eval_only = True
            
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)

        only_id = 'all'
        if cfg.train.max_tasks == 1 and not cfg.train.eval_only_all:
            task_order = cfg.train.task_order
            if cfg.train.task_order not in (None, -1, [], 'None', '', '()', '[]', '-1'):
                if isinstance(task_order, str):
                    task_order = [int(i.strip()) for i in task_order.strip('[]').strip('()').split(',')]
                    only_id = task_order[0]
        if cfg.train.global_ignore_class_ids not in (None, -1, [], 'None', '', '()', '[]', '-1'):
            global_ignore_class_ids = [int(i.strip()) for i in cfg.train.global_ignore_class_ids.strip('[]').strip('()').split(',')]
        else:
            global_ignore_class_ids = None

        print(do_test(cfg, model, eval_only=True,
                      mode='test', only_id=0,
                      json_name=f'eval_task_{only_id}_test',
                      global_ignore_class_ids=global_ignore_class_ids, 
                      eval_single_class=True,
                      task_id=cfg.train.max_task_id_eval))
    else:
        do_train(args, cfg)
    
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
