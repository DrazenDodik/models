# DeepLab on Valohai :shark:

Below a short summary of how the DeepLab sample was brought to Valohai.

* Remember to check out [Valohai Quickstart](https://docs.valohai.com/tutorials/valohai/) and [Valohai Quickstart - Advanced Topics](https://docs.valohai.com/tutorials/valohai/advanced/) for a more detailed explanation of the core Valohai concepts.

Other good docs.valohai.com reads are
* https://docs.valohai.com/executions/inputs/
* https://docs.valohai.com/executions/outputs/
* https://docs.valohai.com/executions/live-outputs/
* https://docs.valohai.com/valohai-yaml/

## Install Valohai CLI tools

* `pip install valohai-cli` and login using your credentials (`vh login`)
* In the root of the project initialize a new project and link your current working directory to that Valohai project (`vh init`)
    * Select `tensorflow/tensorflow:1.15.4-gpu-py3` as the Docker image for the project
    * Create a new Valohai project and link your directory to that project

## Load and convert the ADE20K Dataset
In our example we'll be using the ADE20K dataset as an example. We'll create a Valohai step that downloads the data from the cloud, and then runs `datasets/build_ade20k_data.py` to extract and convert the dataset to TFRecords that we can use in training.

Before continuing dowload the ADE20K sample dataset and upload it to your cloud storage. Go to app.valohai.com -> MyProject -> Data -> Upload. Click on the action button to get a "Copy datum:// URI" or "Copy azure:// URI" action. You can use either one of these as your default input for the YAML below. (Note: the Azure link is available only, if you've configured your private Azure data store)

* Create a new step in `valohai.yaml` that uses the `tensorflow:1.15.4-gpu-py3` Docker image. This step should have a single input (the raw dataset, currently stored in our sample Azure Blob Storage) and parameters that are defined in the `build_ade20k_data.py`
    * The step below is running two commands: **1)** Extract the downloaded zip file **2)** Run `build_ade20k_data.py` with the relevant parameters. These commands will be ran at the beginning of each execution.
    ```yaml
    - step:
        name: Load data and convert
        description: Converts ADE20K data to TFRecord file format with Example protos
        image: tensorflow/tensorflow:1.15.4-gpu-py3
        command:
            - unzip /valohai/inputs/ADE20K/ADEChallengeData2016.zip -d ADE20K/
            - python research/deeplab/datasets/build_ade20k_data.py {parameters}
        inputs:
        - name: ADE20K
            default: azure://tcsvalohai/deeplab/data/01ERE/01EREM82KQF8DNWGJT3GNM3Q35/upload/ADEChallengeData2016.zip
        parameters:
        - name: train_image_folder
            type: string
            default: "./ADE20K/ADEChallengeData2016/images/training"
        - name: train_image_label_folder
            type: string
            default: "./ADE20K/ADEChallengeData2016/annotations/training"
        - name: val_image_folder
            type: string
            default: "./ADE20K/ADEChallengeData2016/images/validation"
        - name: val_image_label_folder
            type: string
            default: "./ADE20K/ADEChallengeData2016/annotations/validation"
        - name: output_dir
            type: string
            default: "./ADE20K/tfrecord"
    ``` 
* Open `research/deeplab/datasets/build_ade20k_data.py`.    
    * Add `import tarfile` and update `main()` with the following
    * ```python
        # VALOHAI: 
        # Create a tar.gz in the outputs directory.
        # Get the location of the outputs directory by reading th environment variable that is passed to all machines (VH_OUTPUTS_DIR)
        # Create a compressed file with all the generated tfrecords
        # The compressed file will be placed to the Valohai outputs directory
        # From there it will get automatically uploaded to the cloud so you can use the generated files in other executions

        OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR')

        tfrecords = tarfile.open(os.path.join(OUTPUTS_DIR, 'tfrecords.tar.gz'), 'w:gz')
        for dirpath, dirnames, filenames in os.walk(FLAGS.output_dir):
            for filename in filenames:
                filepath   = os.path.join(dirpath, filename)
                tfrecords.add(filepath, filename)
        tfrecords.close()
        ```
* Now you can run the execution in Valohai as an ad-hoc execution through the command-line: `vh exec run load --adhoc`.
    * The `load` points to the `step.name` in `valohai.yaml` (Load data and convert). It's doing a simple substring, and as there is only one step starting with load, we can just type load instead of using the full name.
    * `--adhoc` means that Valohai shouldn't go to GitHub to fetch a commit that will be used to run this, instead the CLI should package everything in this folder (`models`) and send it to Valohai for an execution. This is useful for quick testing, but ultimately you should commit and push to your Git repository.

Head on over to app.valohai.com and see the execution running. Once it's done, you'll see the zip file appear in the outputs tab. Head ot the outputs-tab and click on the button to copy the datum link to the file (or copy the azure:// link)

## Train a model

We'll update the model training script to download the TFRecords from Azure Blob Storage (the tar.gz we saved in the previous step), define a set of Valohai parameters and print our key metrics (e.g loss).

* Let's start by creating a new step called `Train DeepLab model` in `valohai.yaml`. 
    * This model requires some additional libraries that are not included as a part of the `tensorflow/tensorflow:1.15` Docker image, so we're running `pip install -r requirements.txt` to install missing libraries.
    * We're also updating the PYTHONPATH to contain the modules from `research/slim` as those will be used in DeepLab.
    * ```yaml
        - step:
            name: Train DeepLab model
            image: tensorflow/tensorflow:1.15.4-gpu-py3
            command:
                - export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research/slim:`pwd`/research
                - pip install -r requirements.txt
        ```
    * The `requirements.txt` file is created in the root of the folder and just contains `tf-slim==1.1.0`
* Our `train.py` will need some data to train the model, so we'll need to provide the `tfrecords` generated in the previous step as an input to this step.
    * Add a inputs section for the `Train DeepLab model` step.
    * Replace the default address with the datum link or Azure link you copied from the previous steps outputs-tab.
        ```yaml
        inputs:
          - name: tfrecords
            default: azure://tcsvalohai/deeplab/data/01ERK/01ERKQZ902916DJZ72FXRS60Y4/output-137/tfrecords.tar.gz
        ```
* Our `train.py` expects to find a folder with tfrecords, not a tar.gz file that we're currently offering as a input. So let's create a new command to extract the compressed file, before running the train.py script.
    * ```yaml
        - step:
            name: Train DeepLab model
            image: tensorflow/tensorflow:1.15.4-gpu-py3
            command:
                - export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research/slim:`pwd`/research
                - pip install -r requirements.txt
                - mkdir tfrecords
                - tar -xvzf /valohai/inputs/tfrecords/tfrecords.tar.gz -C /valohai/repository/tfrecords/
                - python research/deeplab/train.py --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 {parameters}
            inputs:
            - name: tfrecords
                default: azure://tcsvalohai/deeplab/data/01ERK/01ERKQZ902916DJZ72FXRS60Y4/output-137/tfrecords.tar.gz
        ```
* Finally, in the same way we defined inputs to this steps, we want to define a set of parameters that you can change and optimize through Valohai. The parameters listing in based on the TensorFlow sample. You can just copy&paste the definition.
    * ```yaml
        - step:
            name: Train DeepLab model
            image: tensorflow/tensorflow:1.15.4-gpu-py3
            command:
                - export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research/slim:`pwd`/research
                - pip install -r requirements.txt
                - mkdir tfrecords
                - tar -xvzf /valohai/inputs/tfrecords/tfrecords.tar.gz -C /valohai/repository/tfrecords/
                - python research/deeplab/train.py --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 {parameters}
            inputs:
            - name: tfrecords
                default: azure://tcsvalohai/deeplab/data/01ERK/01ERKQZ902916DJZ72FXRS60Y4/output-137/tfrecords.tar.gz
            parameters:
            - name: logtostderr
                type: flag
                default: True
                pass-as: --logtostderr={v}
            - name: training_number_of_steps
                type: integer
                default: 100
                description: "The number of steps used for training"
            - name: train_split
                type: string
                default: "train"
                description: "Which split of the dataset to be used for training"
            - name: model_variant
                type: string
                default: "xception_65"
            - name: output_stride
                type: integer
                default: 16
            - name: decoder_output_stride
                type: integer
                default: 4
            - name: train_crop_size
                type: string
                default: "513,513"
                description: "Image crop size [height, width] during training."
            - name: train_batch_size
                type: integer
                default: 4
                description: "The number of images in each batch during training."
            - name: min_resize_value
                type: integer
                default: 513
            - name: max_resize_value
                type: integer
                default: 513
            - name: resize_factor
                type: integer
                default: 16
            - name: dataset
                type: string
                default: "ade20k"
                description: "Name of the segmentation dataset."
            - name: dataset_dir
                type: string
                default: "./tfrecords"
            - name: train_logdir
                type: string
                default: "/valohai/repository/checkpoints"
        ```
    * Now to pass these parameters to our Python script, we need to change the command to include a placeholder for all parameters:
        * `python research/deeplab/train.py --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 {parameters}`
        * Valohai will replace `{parameters}` with the parameters that are passed to the execution.
        * Valohai doesn't currently support lists as paramters, so I've left them as in the sample. Alternatively you could just pass them as string and then split them to a list in your Python.

:arrow_right: Read more about Valohai parameters at https://docs.valohai.com/core-concepts/parameters/

You can now run the step on valohai with `vh exec run train --adhoc`
* Note you can go the Valohai UI and copy an existing execution, and change the parameters directly in the UI (or copy an existing Execution as a Task for parameter sweeps)

Next we can go and update `research/deeplab/train.py` with some Valohai specific lines.

Valohai can collect key metrics from your executions (e.g. accuracy, loss etc.) as long as this is printed out as JSON during the execution. By default the TF Sample doesn't print out JSON, so we need to create a custom train_step function that will be used when training the model. In it, we can add the JSON printing.

* Add `import json` to `train.py` as we'll need it when printing Valohai metadata
* Edit `slim.learning.train` in side your `main()`
    ```python
          slim.learning.train(
          train_tensor,
          train_step_fn=train_step, #Added a custom train_step_function
          logdir=FLAGS.train_logdir,
          log_every_n_steps=FLAGS.log_steps,
          master=FLAGS.master,
          number_of_steps=FLAGS.training_number_of_steps,
          is_chief=(FLAGS.task == 0),
          chief_only_hooks=your_hooks,
          session_config=session_config,
          startup_delay_steps=startup_delay_steps,
          init_fn=init_fn,
          summary_op=summary_op,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)
    ```
* Now let's create that function, so it can be actually called :sweat_smile:
* Our sample below just collects the `loss` metric, but you'd do the same approach for other metrics
```python
# https://github.com/google-research/tf-slim/blob/master/tf_slim/learning.py
def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the total
      loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(
          run_metadata, 'run_metadata-%d' % np_global_step)

  # In addition to using logging.info to log the loss for each step, we'll want to print out JSON that Valohai can read as metadata.
  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      # If it's JSON, it will be available as metadata in Valohai
      # You can then sort and compare different runs using these metrics.
      print(json.dumps({
        "step": str(np_global_step),
        "loss": str(total_loss)
      }))
      logging.info('global step %d: loss = %.4f (%.3f sec/step)', np_global_step, total_loss, time_elapsed)

  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop

```

You can now run the step on valohai with `vh exec run train --adhoc`
* You'll notice the `metadata` tab of an execution activate. Choose step on the x-axis and loss to the y-axis to see the values develop over time.
* You can also see the metrics (and parameters) on the "Executions" table by clicking the "Show Columns" button and enabling the metrics you want to see.

:arrow_right: Read more about collecting metadata with Valohai at https://docs.valohai.com/executions/metadata/ 


## Export model to frozen graph

Let's start by adding a new step to our `valohai.yaml`. This step will export a trained TF model to a frozen graph.

The step is quite familiar to the previous steps:
* Define the name, Docker image, commands to execute, data inputs and parameters
* In the inputs, we're taking in the `checkpoints.tar.gz` generated in the previous step.
* Under commands we against install any additional libraries we might need, extract the downloaded file and run the `export_model.py`

```yaml
- step:
    name: Export trained model to TF frozen graph
    image: tensorflow/tensorflow:1.15.4-gpu-py3
    command:
        - export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research/slim:`pwd`/research
        - pip install -r requirements.txt
        - mkdir checkpoints
        - tar -xvzf /valohai/inputs/checkpoints/checkpoints.tar.gz -C /valohai/repository/checkpoints/
        - python research/deeplab/export_model.py --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 {parameters}
    inputs:
      - name: checkpoints
        default: azure://tcsvalohai/deeplab/data/01ERK/01ERKQZ902916DJZ72FXRS60Y4/output-141/checkpoints.tar.gz
    parameters:
      - name: dataset
        type: string
        default: "ade20k"
        description: "Name of the segmentation dataset."
      - name: checkpoint_path
        type: string
        default: "/valohai/repository/checkpoints/"
      - name: save_inference_graph
        type: flag
        default: True
      - name: export_path
        type: string
        default: "/valohai/outputs/export"
      - name: model_variant
        type: string
        default: "xception_65"
        description: "Name of the segmentation dataset."
      - name: output_stride
        type: integer
        default: 8
      - name: num_classes
        type: integer
        default: 151
```

* You can update the `export_model.py` to read the export path of a single checkpoint. In `export_model.py`
    ```python
        freeze_graph.freeze_graph_with_def_protos(
            graph_def,
            saver.as_saver_def(),
            tf.train.latest_checkpoint(FLAGS.checkpoint_path), # VALOHAI: Updated to fetch the latest checkpoint
            _OUTPUT_NAME + ',' + _OUTPUT_PROB_NAME,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=FLAGS.export_path,
            clear_devices=True,
            initializer_nodes=None)
    ```

## Inference

Valohai Deployments can be used for online inference, which gives you an online REST endpoint that you can call and get a response. In this case you could have an endpoint to which you send an image and get the model predictions as a result.

If you don't have a need for online prediction, with getting results immediately, you can also use standard Valohai executions to perform batch executions. You'd start an execution that takes a number of image files as an input, performs predictions on them and outputs the results.

The `valohai.yaml` also contains an example pipeline that chains the two steps together. 

:arrow_right: Read more about pipelines here https://docs.valohai.com/core-concepts/pipelines/

# DeepLab: Deep Labelling for Semantic Image Segmentation

DeepLab is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g., person, dog, cat and so on)
to every pixel in the input image. Current implementation includes the following
features:

1.  DeepLabv1 [1]: We use *atrous convolution* to explicitly control the
    resolution at which feature responses are computed within Deep Convolutional
    Neural Networks.

2.  DeepLabv2 [2]: We use *atrous spatial pyramid pooling* (ASPP) to robustly
    segment objects at multiple scales with filters at multiple sampling rates
    and effective fields-of-views.

3.  DeepLabv3 [3]: We augment the ASPP module with *image-level feature* [5, 6]
    to capture longer range information. We also include *batch normalization*
    [7] parameters to facilitate the training. In particular, we applying atrous
    convolution to extract output features at different output strides during
    training and evaluation, which efficiently enables training BN at output
    stride = 16 and attains a high performance at output stride = 8 during
    evaluation.

4.  DeepLabv3+ [4]: We extend DeepLabv3 to include a simple yet effective
    decoder module to refine the segmentation results especially along object
    boundaries. Furthermore, in this encoder-decoder structure one can
    arbitrarily control the resolution of extracted encoder features by atrous
    convolution to trade-off precision and runtime.

If you find the code useful for your research, please consider citing our latest
works:

*   DeepLabv3+:

```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

*   MobileNetv2:

```
@inproceedings{mobilenetv22018,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  booktitle={CVPR},
  year={2018}
}
```

*   MobileNetv3:

```
@inproceedings{mobilenetv32019,
  title={Searching for MobileNetV3},
  author={Andrew Howard and Mark Sandler and Grace Chu and Liang-Chieh Chen and Bo Chen and Mingxing Tan and Weijun Wang and Yukun Zhu and Ruoming Pang and Vijay Vasudevan and Quoc V. Le and Hartwig Adam},
  booktitle={ICCV},
  year={2019}
}
```

*  Architecture search for dense prediction cell:

```
@inproceedings{dpc2018,
  title={Searching for Efficient Multi-Scale Architectures for Dense Image Prediction},
  author={Liang-Chieh Chen and Maxwell D. Collins and Yukun Zhu and George Papandreou and Barret Zoph and Florian Schroff and Hartwig Adam and Jonathon Shlens},
  booktitle={NIPS},
  year={2018}
}

```

*  Auto-DeepLab (also called hnasnet in core/nas_network.py):

```
@inproceedings{autodeeplab2019,
  title={Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic
Image Segmentation},
  author={Chenxi Liu and Liang-Chieh Chen and Florian Schroff and Hartwig Adam
  and Wei Hua and Alan Yuille and Li Fei-Fei},
  booktitle={CVPR},
  year={2019}
}

```


In the current implementation, we support adopting the following network
backbones:

1.  MobileNetv2 [8] and MobileNetv3 [16]: A fast network structure designed
    for mobile devices.

2.  Xception [9, 10]: A powerful network structure intended for server-side
    deployment.

3.  ResNet-v1-{50,101} [14]: We provide both the original ResNet-v1 and its
    'beta' variant where the 'stem' is modified for semantic segmentation.

4.  PNASNet [15]: A Powerful network structure found by neural architecture
    search.

5.  Auto-DeepLab (called HNASNet in the code): A segmentation-specific network
    backbone found by neural architecture search.

This directory contains our TensorFlow [11] implementation. We provide codes
allowing users to train the model, evaluate results in terms of mIOU (mean
intersection-over-union), and visualize segmentation results. We use PASCAL VOC
2012 [12] and Cityscapes [13] semantic segmentation benchmarks as an example in
the code.

Some segmentation results on Flickr images:
<p align="center">
    <img src="g3doc/img/vis1.png" width=600></br>
    <img src="g3doc/img/vis2.png" width=600></br>
    <img src="g3doc/img/vis3.png" width=600></br>
</p>

## Contacts (Maintainers)

*   Liang-Chieh Chen, github: [aquariusjay](https://github.com/aquariusjay)
*   YuKun Zhu, github: [yknzhu](https://github.com/YknZhu)
*   George Papandreou, github: [gpapan](https://github.com/gpapan)
*   Hui Hui, github: [huihui-personal](https://github.com/huihui-personal)
*   Maxwell D. Collins, github: [mcollinswisc](https://github.com/mcollinswisc)
*   Ting Liu: github: [tingliu](https://github.com/tingliu)

## Tables of Contents

Demo:

*   <a href='https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb'>Colab notebook for off-the-shelf inference.</a><br>

Running:

*   <a href='g3doc/installation.md'>Installation.</a><br>
*   <a href='g3doc/pascal.md'>Running DeepLab on PASCAL VOC 2012 semantic segmentation dataset.</a><br>
*   <a href='g3doc/cityscapes.md'>Running DeepLab on Cityscapes semantic segmentation dataset.</a><br>
*   <a href='g3doc/ade20k.md'>Running DeepLab on ADE20K semantic segmentation dataset.</a><br>

Models:

*   <a href='g3doc/model_zoo.md'>Checkpoints and frozen inference graphs.</a><br>

Misc:

*   Please check <a href='g3doc/faq.md'>FAQ</a> if you have some questions before reporting the issues.<br>

## Getting Help

To get help with issues you may encounter while using the DeepLab Tensorflow
implementation, create a new question on
[StackOverflow](https://stackoverflow.com/) with the tag "tensorflow".

Please report bugs (i.e., broken code, not usage questions) to the
tensorflow/models GitHub [issue
tracker](https://github.com/tensorflow/models/issues), prefixing the issue name
with "deeplab".

## License

All the codes in deeplab folder is covered by the [LICENSE](https://github.com/tensorflow/models/blob/master/LICENSE)
under tensorflow/models. Please refer to the LICENSE for details.

## Change Logs

### March 26, 2020
* Supported EdgeTPU-DeepLab and EdgeTPU-DeepLab-slim on Cityscapes.
**Contributor**: Yun Long.

### November 20, 2019
* Supported MobileNetV3 large and small model variants on Cityscapes.
**Contributor**: Yukun Zhu.


### March 27, 2019

* Supported using different loss weights on different classes during training.
**Contributor**: Yuwei Yang.


### March 26, 2019

* Supported ResNet-v1-18. **Contributor**: Michalis Raptis.


### March 6, 2019

* Released the evaluation code (under the `evaluation` folder) for image
parsing, a.k.a. panoptic segmentation. In particular, the released code supports
evaluating the parsing results in terms of both the parsing covering and
panoptic quality metrics. **Contributors**: Maxwell Collins and Ting Liu.


### February 6, 2019

* Updated decoder module to exploit multiple low-level features with different
output_strides.

### December 3, 2018

* Released the MobileNet-v2 checkpoint on ADE20K.


### November 19, 2018

* Supported NAS architecture for feature extraction. **Contributor**: Chenxi Liu.

* Supported hard pixel mining during training.


### October 1, 2018

* Released MobileNet-v2 depth-multiplier = 0.5 COCO-pretrained checkpoints on
PASCAL VOC 2012, and Xception-65 COCO pretrained checkpoint (i.e., no PASCAL
pretrained).


### September 5, 2018

* Released Cityscapes pretrained checkpoints with found best dense prediction cell.


### May 26, 2018

* Updated ADE20K pretrained checkpoint.


### May 18, 2018
* Added builders for ResNet-v1 and Xception model variants.
* Added ADE20K support, including colormap and pretrained Xception_65 checkpoint.
* Fixed a bug on using non-default depth_multiplier for MobileNet-v2.


### March 22, 2018

* Released checkpoints using MobileNet-V2 as network backbone and pretrained on
PASCAL VOC 2012 and Cityscapes.


### March 5, 2018

* First release of DeepLab in TensorFlow including deeper Xception network
backbone. Included chekcpoints that have been pretrained on PASCAL VOC 2012
and Cityscapes.

## References

1.  **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**<br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (+ equal
    contribution). <br />
    [[link]](https://arxiv.org/abs/1412.7062). In ICLR, 2015.

2.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille (+ equal
    contribution). <br />
    [[link]](http://arxiv.org/abs/1606.00915). TPAMI 2017.

3.  **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

4.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

5.  **ParseNet: Looking Wider to See Better**<br />
    Wei Liu, Andrew Rabinovich, Alexander C Berg<br />
    [[link]](https://arxiv.org/abs/1506.04579). arXiv:1506.04579, 2015.

6.  **Pyramid Scene Parsing Network**<br />
    Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia<br />
    [[link]](https://arxiv.org/abs/1612.01105). In CVPR, 2017.

7.  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate shift**<br />
    Sergey Ioffe, Christian Szegedy <br />
    [[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

8.  **MobileNetV2: Inverted Residuals and Linear Bottlenecks**<br />
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />
    [[link]](https://arxiv.org/abs/1801.04381). In CVPR, 2018.

9.  **Xception: Deep Learning with Depthwise Separable Convolutions**<br />
    François Chollet<br />
    [[link]](https://arxiv.org/abs/1610.02357). In CVPR, 2017.

10. **Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry**<br />
    Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, Jifeng Dai<br />
    [[link]](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf). ICCV COCO Challenge
    Workshop, 2017.

11. **Tensorflow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**<br />
    M. Abadi, A. Agarwal, et al. <br />
    [[link]](https://arxiv.org/abs/1603.04467). arXiv:1603.04467, 2016.

12. **The Pascal Visual Object Classes Challenge – A Retrospective,** <br />
    Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John
    Winn, and Andrew Zisserma. <br />
    [[link]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). IJCV, 2014.

13. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br />
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br />
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.

14. **Deep Residual Learning for Image Recognition**<br />
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. <br />
    [[link]](https://arxiv.org/abs/1512.03385). In CVPR, 2016.

15. **Progressive Neural Architecture Search**<br />
    Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy. <br />
    [[link]](https://arxiv.org/abs/1712.00559). In ECCV, 2018.

16. **Searching for MobileNetV3**<br />
    Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. <br />
    [[link]](https://arxiv.org/abs/1905.02244). In ICCV, 2019.
