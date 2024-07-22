/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.tensorflow.v2.api030;

import com.google.gson.Gson;
import com.google.protobuf.InvalidProtocolBufferException;

import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.tensorflow.v2.api030.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensorflow.v2.api030.tensor.TensorBuilder;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.ZipUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.types.family.TType;

/**
 * Class to that communicates with the dl-model runner, see 
 * @see <a href="https://github.com/bioimage-io/model-runner-java">dlmodelrunner</a>
 * to execute Tensorflow 2 models. This class is compatible with every TF2 Java API
 * but the 0.2.0 version
 * This class implements the interface {@link DeepLearningEngineInterface} to get the 
 * agnostic {@link io.bioimage.modelrunner.tensor.Tensor}, convert them into 
 * {@link org.tensorflow.Tensor}, execute a Tensorflow 2 Deep Learning model on them and
 * convert the results back to {@link io.bioimage.modelrunner.tensor.Tensor} to send them 
 * to the main program in an agnostic manner.
 * 
 * {@link ImgLib2Builder}. Creates ImgLib2 images for the backend
 *  of {@link io.bioimage.modelrunner.tensor.Tensor} from {@link org.tensorflow.Tensor}
 * {@link TensorBuilder}. Converts {@link io.bioimage.modelrunner.tensor.Tensor} into {@link org.tensorflow.Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class Tensorflow2Interface implements DeepLearningEngineInterface {

	private static final String[] MODEL_TAGS = { "serve", "inference", "train",
		"eval", "gpu", "tpu" };

	private static final String[] TF_MODEL_TAGS = {
		"tf.saved_model.tag_constants.SERVING",
		"tf.saved_model.tag_constants.INFERENCE",
		"tf.saved_model.tag_constants.TRAINING",
		"tf.saved_model.tag_constants.EVAL", "tf.saved_model.tag_constants.GPU",
		"tf.saved_model.tag_constants.TPU" };

	private static final String[] SIGNATURE_CONSTANTS = { "serving_default",
		"inputs", "tensorflow/serving/classify", "classes", "scores", "inputs",
		"tensorflow/serving/predict", "outputs", "inputs",
		"tensorflow/serving/regress", "outputs", "train", "eval",
		"tensorflow/supervised/training", "tensorflow/supervised/eval" };

	private static final String[] TF_SIGNATURE_CONSTANTS = {
		"tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY",
		"tf.saved_model.signature_constants.CLASSIFY_INPUTS",
		"tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME",
		"tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES",
		"tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES",
		"tf.saved_model.signature_constants.PREDICT_INPUTS",
		"tf.saved_model.signature_constants.PREDICT_METHOD_NAME",
		"tf.saved_model.signature_constants.PREDICT_OUTPUTS",
		"tf.saved_model.signature_constants.REGRESS_INPUTS",
		"tf.saved_model.signature_constants.REGRESS_METHOD_NAME",
		"tf.saved_model.signature_constants.REGRESS_OUTPUTS",
		"tf.saved_model.signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY",
		"tf.saved_model.signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY",
		"tf.saved_model.signature_constants.SUPERVISED_TRAIN_METHOD_NAME",
		"tf.saved_model.signature_constants.SUPERVISED_EVAL_METHOD_NAME" };
	/**
	 * Name without vesion of the JAR created for this library
	 */
	private static final String JAR_FILE_NAME = "dl-modelrunner-tensorflow-";

	private static final String NAME_KEY = "name";
	private static final String SHAPE_KEY = "shape";
	private static final String DTYPE_KEY = "dtype";
	private static final String IS_INPUT_KEY = "isInput";
	private static final String MEM_NAME_KEY = "memoryName";
	
	private List<SharedMemoryArray> shmaInputList = new ArrayList<SharedMemoryArray>();
	
	private List<SharedMemoryArray> shmaOutputList = new ArrayList<SharedMemoryArray>();
	
	private List<String> shmaNamesList = new ArrayList<String>();

    /**
     * The loaded Tensorflow 2 model
     */
	private static SavedModelBundle model;
	/**
	 * Internal object of the Tensorflow model
	 */
	private static SignatureDef sig;
	/**
	 * Whether the execution needs interprocessing (MacOS Interl) or not
	 */
	private boolean interprocessing = false;
    /**
     * Folde containing the model that is being executed
     */
    private String modelFolder;
    /**
     * Process where the model is being loaded and executed
     */
    Service runner;
	
    public Tensorflow2Interface(boolean doInterprocessing) throws IOException, URISyntaxException
    {
		interprocessing = doInterprocessing;
		if (this.interprocessing) {
			runner = getRunner();
		}
    }
    
    public Tensorflow2Interface() throws IOException, URISyntaxException
    {
		this(true);
    }
    
    private Service getRunner() throws IOException, URISyntaxException {
		List<String> args = getProcessCommandsWithoutArgs();
		String[] argArr = new String[args.size()];
		args.toArray(argArr);

		return new Service(new File(argArr[0]), argArr);
    }

    /**
     * {@inheritDoc}
     * 
     * Load a Tensorflow 2 model. If the machine where the code is
     * being executed is a MacOS Intel or Windows, the model will be loaded in 
     * a separate process each time the method {@link #run(List, List)}
     * is called 
     */
	@Override
	public void loadModel(String modelFolder, String modelSource)
		throws LoadModelException
	{
		this.modelFolder = modelFolder;
		if (interprocessing) {
			try {
				launchModelLoadOnProcess();
			} catch (IOException | InterruptedException e) {
				throw new LoadModelException(Types.stackTrace(e));
			}
			return;
		}
		try {
			checkModelUnzipped();
		} catch (Exception e) {
			throw new LoadModelException(e.toString());
		}
		model = SavedModelBundle.load(this.modelFolder, "serve");
		byte[] byteGraph = model.metaGraphDef().toByteArray();
		try {
			sig = MetaGraphDef.parseFrom(byteGraph).getSignatureDefOrThrow(
				"serving_default");
		}
		catch (InvalidProtocolBufferException e) {
			System.out.println("Invalid graph");
		}
	}
	
	private void launchModelLoadOnProcess() throws IOException, InterruptedException {
		HashMap<String, Object> args = new HashMap<String, Object>();
		args.put("mdoelFolder", modelFolder);
		Task task = runner.task("loadModel", args);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException();
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException();
		else if (task.status == TaskStatus.CRASHED)
			throw new RuntimeException();
	}
	
	/**
	 * Check if an unzipped tensorflow model exists in the model folder, 
	 * and if not look for it and unzip it
	 * @throws LoadModelException if no model is found
	 * @throws IOException if there is any error unzipping the model
	 * @throws Exception if there is any error related to model packaging
	 */
	private void checkModelUnzipped() throws LoadModelException, IOException, Exception {
		if (new File(modelFolder, "variables").isDirectory()
				&& new File(modelFolder, "saved_model.pb").isFile())
			return;
		unzipTfWeights(ModelDescriptor.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME));
	}
	
	/**
	 * Method that unzips the tensorflow model zip into the variables
	 * folder and .pb file, if they are saved in a zip
	 * @throws LoadModelException if not zip file is found
	 * @throws IOException if there is any error unzipping
	 */
	private void unzipTfWeights(ModelDescriptor descriptor) throws LoadModelException, IOException {
		if (new File(modelFolder, "tf_weights.zip").isFile()) {
			System.out.println("Unzipping model...");
			ZipUtils.unzipFolder(modelFolder + File.separator + "tf_weights.zip", modelFolder);
		} else if ( descriptor.getWeights().getAllSuportedWeightNames()
				.contains(EngineInfo.getBioimageioTfKey()) ) {
			String source = descriptor.getWeights().gettAllSupportedWeightObjects().stream()
					.filter(ww -> ww.getFramework().equals(EngineInfo.getBioimageioTfKey()))
					.findFirst().get().getSource();
			source = DownloadModel.getFileNameFromURLString(source);
			System.out.println("Unzipping model...");
			ZipUtils.unzipFolder(modelFolder + File.separator + source, modelFolder);
		} else {
			throw new LoadModelException("No model file was found in the model folder");
		}
	}
	/**
	 * {@inheritDoc}
	 * 
	 * Run a Tensorflow2 model on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 */
	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors)
		throws RunModelException
	{
		if (interprocessing) {
			runInterprocessing(inputTensors, outputTensors);
			return;
		}
		Session session = model.session();
		Session.Runner runner = session.runner();
		List<String> inputListNames = new ArrayList<String>();
		List<TType> inTensors = new ArrayList<TType>();
		int c = 0;
		for (Tensor<?> tt : inputTensors) {
			inputListNames.add(tt.getName());
			TType inT = TensorBuilder.build(tt);
			inTensors.add(inT);
			String inputName = getModelInputName(tt.getName(), c ++);
			runner.feed(inputName, inT);
		}
		c = 0;
		for (Tensor<?> tt : outputTensors)
			runner = runner.fetch(getModelOutputName(tt.getName(), c ++));
		// Run runner
		List<org.tensorflow.Tensor> resultPatchTensors = runner.run();

		// Fill the agnostic output tensors list with data from the inference result
		fillOutputTensors(resultPatchTensors, outputTensors);
		// Close the remaining resources
		session.close();
		for (TType tt : inTensors) {
			tt.close();
		}
		for (org.tensorflow.Tensor tt : resultPatchTensors) {
			tt.close();
		}
	}
	
	protected void runFromShmas(LinkedHashMap<String, Object> inputs, LinkedHashMap<String, Object> outputs) {
		Session session = model.session();
		Session.Runner runner = session.runner();
		
		List<TType> inTensors = new ArrayList<TType>();
		int c = 0;
		for (Entry<String, Object> ee : inputs.entrySet()) {
			Map<String, Object> decoded = Types.decode((String) ee.getValue());
			SharedMemoryArray shma = SharedMemoryArray.read((String) decoded.get(MEM_NAME_KEY));
			TType inT = io.bioimage.modelrunner.tensorflow.v2.api030.shm.TensorBuilder.build(shma);
			inTensors.add(inT);
			String inputName = getModelInputName(ee.getKey(), c ++);
			runner.feed(inputName, inT);
		}
		
		c = 0;
		for (Entry<String, Object> ee : outputs.entrySet())
			runner = runner.fetch(getModelOutputName(ee.getKey(), c ++));
		// Run runner
		List<org.tensorflow.Tensor> resultPatchTensors = runner.run();

		// Fill the agnostic output tensors list with data from the inference result
		for (Entry<String, Object> ee : outputs.entrySet()) {
			Map<String, Object> decoded = Types.decode((String) ee.getValue());
			SharedMemoryArray shma = SharedMemoryArray.read((String) decoded.get(MEM_NAME_KEY));
			TType inT = io.bioimage.modelrunner.tensorflow.v2.api030.shm.TensorBuilder.build(shma);
			inTensors.add(inT);
			String inputName = getModelInputName(ee.getKey(), c ++);
			runner.feed(inputName, inT);
		}
		// Close the remaining resources
		session.close();
		for (TType tt : inTensors) {
			tt.close();
		}
		for (org.tensorflow.Tensor tt : resultPatchTensors) {
			tt.close();
		}
	}
	
	/**
	 * MEthod only used in MacOS Intel and Windows systems that makes all the arrangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public void runInterprocessing(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		shmaInputList = new ArrayList<SharedMemoryArray>();
		shmaOutputList = new ArrayList<SharedMemoryArray>();
		LinkedHashMap<String, String> encIns = encodeInputs(inputTensors);
		modifyForWinCmd(encIns);
		LinkedHashMap<String, String> encOuts = encodeOutputs(outputTensors);
		modifyForWinCmd(encOuts);
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		args.put("inputs", encIns);
		args.put("outputs", encOuts);

		try {
			Task task = runner.task("inference", args);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.CRASHED)
				throw new RuntimeException();
			for (int i = 0; i < outputTensors.size(); i ++) {
	        	String name = (String) Types.decode(encOuts.get(outputTensors.get(i).getName())).get(MEM_NAME_KEY);
	        	SharedMemoryArray shm = shmaOutputList.stream()
	        			.filter(ss -> ss.getName().equals(name)).findFirst().orElse(null);
	        	if (shm == null) {
	        		shm = SharedMemoryArray.read(name);
	        		shmaOutputList.add(shm);
	        	}
	        	RandomAccessibleInterval<?> rai = shm.getSharedRAI();
	        	outputTensors.get(i).setData(Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai), Util.getTypeFromInterval(Cast.unchecked(rai))));
	        }
		} catch (Exception e) {
			closeShmas();
			if (e instanceof RunModelException)
				throw (RunModelException) e;
			throw new RunModelException(Types.stackTrace(e));
		}
		closeShmas();
	}
	
	private void closeShmas() {
		shmaInputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaInputList = null;
		shmaOutputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaOutputList = null;
	}
	
	private static void modifyForWinCmd(LinkedHashMap<String, String> ins){
		if (!PlatformDetection.isWindows())
			return;
		ins.entrySet().forEach(ee ->{
			String val = ee.getValue();
			String nVal = "\"" + val.replace("\"", "\\\"") + "\"";
			ee.setValue(nVal);
		});
	}
	
	
	private LinkedHashMap<String, String> encodeInputs(List<Tensor<?>> inputTensors) {
		LinkedHashMap<String, String> encodedInputTensors = new LinkedHashMap<String, String>();
		Gson gson = new Gson();
		for (Tensor<?> tt : inputTensors) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
			shmaInputList.add(shma);
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(SHAPE_KEY, tt.getShape());
			map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
			map.put(IS_INPUT_KEY, true);
			map.put(MEM_NAME_KEY, shma.getName());
			encodedInputTensors.put(tt.getName(), gson.toJson(map));
		}
		return encodedInputTensors;
	}
	
	
	private LinkedHashMap<String, String> encodeOutputs(List<Tensor<?>> outputTensors) {
		Gson gson = new Gson();
		LinkedHashMap<String, String>encodedOutputTensors = new LinkedHashMap<String, String>();
		for (Tensor<?> tt : outputTensors) {
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(IS_INPUT_KEY, false);
			if (!tt.isEmpty()) {
				map.put(SHAPE_KEY, tt.getShape());
				map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
				SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else if (PlatformDetection.isWindows()){
				SharedMemoryArray shma = SharedMemoryArray.create(0);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else {
				String memName = SharedMemoryArray.createShmName();
				map.put(MEM_NAME_KEY, memName);
				shmaNamesList.add(memName);
			}
			encodedOutputTensors.put(tt.getName(), gson.toJson(map));
		}
		return encodedOutputTensors;
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by the dl-modelrunner
	 * 
	 * @param outputTfTensors an List containing dl-modelrunner tensors
	 * @param outputTensors the names given to the tensors by the model
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static void fillOutputTensors(
			List<org.tensorflow.Tensor> outputTfTensors, List<Tensor<?>> outputTensors)
			throws RunModelException
		{
			if (outputTfTensors.size() != outputTensors.size())
				throw new RunModelException(outputTfTensors.size(), outputTensors.size());
			for (int i = 0; i < outputTfTensors.size(); i++) {
				outputTensors.get(i).setData(ImgLib2Builder.build((TType) outputTfTensors
					.get(i)));
			}
		}

	/**
	 * {@inheritDoc}
	 * 
	 * Close the Tensorflow 2 {@link #model} and {@link #sig}. For 
	 * MacOS Intel and Windows systems it also deletes the temporary files created to
	 * communicate with the other process
	 */
	@Override
	public void closeModel() {
		if (this.interprocessing && runner != null) {
			this.runner.close();
			return;
		} else if (this.interprocessing) {
			return;
		}
		sig = null;
		if (model != null) {
			model.session().close();
			model.close();
		}
		model = null;
	}

	// TODO make only one
	/**
	 * Retrieves the readable input name from the graph signature definition given
	 * the signature input name.
	 * 
	 * @param inputName Signature input name.
	 * @param i position of the input of interest in the list of inputs
	 * @return The readable input name.
	 */
	public static String getModelInputName(String inputName, int i) {
		TensorInfo inputInfo = sig.getInputsMap().getOrDefault(inputName, null);
		if (inputInfo == null) {
			inputInfo = sig.getInputsMap().values().stream().collect(Collectors.toList()).get(i);
		}
		if (inputInfo != null) {
			String modelInputName = inputInfo.getName();
			if (modelInputName != null) {
				if (modelInputName.endsWith(":0")) {
					return modelInputName.substring(0, modelInputName.length() - 2);
				}
				else {
					return modelInputName;
				}
			}
			else {
				return inputName;
			}
		}
		return inputName;
	}

	/**
	 * Retrieves the readable output name from the graph signature definition
	 * given the signature output name.
	 * 
	 * @param outputName Signature output name.
	 * @param i position of the input of interest in the list of inputs
	 * @return The readable output name.
	 */
	public static String getModelOutputName(String outputName, int i) {
		TensorInfo outputInfo = sig.getOutputsMap().getOrDefault(outputName, null);
		if (outputInfo == null) {
			outputInfo = sig.getOutputsMap().values().stream().collect(Collectors.toList()).get(i);
		}
		if (outputInfo != null) {
			String modelOutputName = outputInfo.getName();
			if (modelOutputName.endsWith(":0")) {
				return modelOutputName.substring(0, modelOutputName.length() - 2);
			}
			else {
				return modelOutputName;
			}
		}
		else {
			return outputName;
		}
	}
	
	
	/**
	 * Methods to run interprocessing and bypass the errors that occur in MacOS intel
	 * with the compatibility between TF2 and TF1/Pytorch
	 * This method checks that the arguments are correct, retrieves the input and output
	 * tensors, loads the model, makes inference with it and finally sends the tensors
	 * to the original process
     * 
     * @param args
     * 	arguments of the program:
     * 		- Path to the model folder
     * 		- Path to a temporary dir
     * 		- Name of the input 0
     * 		- Name of the input 1
     * 		- ...
     * 		- Name of the output n
     * 		- Name of the output 0
     * 		- Name of the output 1
     * 		- ...
     * 		- Name of the output n
     * @throws LoadModelException if there is any error loading the model
     * @throws IOException	if there is any error reading or writing any file or with the paths
     * @throws RunModelException	if there is any error running the model
     */
    public static void main(String[] args) throws LoadModelException, IOException, RunModelException {
    }
	
	/**
	 * if java bin dir contains any special char, surround it by double quotes
	 * @param javaBin
	 * 	java bin dir
	 * @return impored java bin dir if needed
	 */
	private static String padSpecialJavaBin(String javaBin) {
		String[] specialChars = new String[] {" "};
        for (String schar : specialChars) {
        	if (javaBin.contains(schar) && PlatformDetection.isWindows()) {
        		return "\"" + javaBin + "\"";
        	}
        }
        return javaBin;
	}
	
	/**
	 * Create the arguments needed to execute tensorflow 2 in another 
	 * process with the corresponding tensors
	 * @return the command used to call the separate process
	 * @throws IOException if the command needed to execute interprocessing is too long
	 * @throws URISyntaxException if there is any error with the URIs retrieved from the classes
	 */
	private List<String> getProcessCommandsWithoutArgs() throws IOException, URISyntaxException {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        ProtectionDomain protectionDomain = Tensorflow2Interface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
        f_name = new File(f_name).getAbsolutePath();
        for (File ff : new File(f_name).getParentFile().listFiles()) {
        	if (ff.getName().startsWith(JAR_FILE_NAME) && !ff.getAbsolutePath().equals(f_name))
        		continue;
        	classpath += ff.getAbsolutePath() + File.pathSeparator;
        }
        String className = Tensorflow2Interface.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        return command;
	}
	
	/**
	 * Method that gets the path to the JAR from where a specific class is being loaded
	 * @param clazz
	 * 	class of interest
	 * @return the path to the JAR that contains the class
	 * @throws UnsupportedEncodingException if the url of the JAR is not encoded in UTF-8
	 */
	private static String getPathFromClass(Class<?> clazz) throws UnsupportedEncodingException {
	    String classResource = clazz.getName().replace('.', '/') + ".class";
	    URL resourceUrl = clazz.getClassLoader().getResource(classResource);
	    if (resourceUrl == null) {
	        return null;
	    }
	    String urlString = resourceUrl.toString();
	    if (urlString.startsWith("jar:")) {
	        urlString = urlString.substring(4);
	    }
	    if (urlString.startsWith("file:/") && PlatformDetection.isWindows()) {
	        urlString = urlString.substring(6);
	    } else if (urlString.startsWith("file:/") && !PlatformDetection.isWindows()) {
	        urlString = urlString.substring(5);
	    }
	    urlString = URLDecoder.decode(urlString, "UTF-8");
	    File file = new File(urlString);
	    String path = file.getAbsolutePath();
	    if (path.lastIndexOf(".jar!") != -1)
	    	path = path.substring(0, path.lastIndexOf(".jar!")) + ".jar";
	    return path;
	}
	
	/**
	 * GEt the directory where the TF2 engine is located if a temporary dir is not found
	 * @return directory of the engines
	 */
	private static String getEnginesDir() {
		String dir;
		try {
			dir = getPathFromClass(Tensorflow2Interface.class);
		} catch (UnsupportedEncodingException e) {
			String classResource = Tensorflow2Interface.class.getName().replace('.', '/') + ".class";
		    URL resourceUrl = Tensorflow2Interface.class.getClassLoader().getResource(classResource);
		    if (resourceUrl == null) {
		        return null;
		    }
		    String urlString = resourceUrl.toString();
		    if (urlString.startsWith("jar:")) {
		        urlString = urlString.substring(4);
		    }
		    if (urlString.startsWith("file:/") && PlatformDetection.isWindows()) {
		        urlString = urlString.substring(6);
		    } else if (urlString.startsWith("file:/") && !PlatformDetection.isWindows()) {
		        urlString = urlString.substring(5);
		    }
		    File file = new File(urlString);
		    String path = file.getAbsolutePath();
		    if (path.lastIndexOf(".jar!") != -1)
		    	path = path.substring(0, path.lastIndexOf(".jar!")) + ".jar";
		    dir = path;
		}
		return new File(dir).getParent();
	}
}
