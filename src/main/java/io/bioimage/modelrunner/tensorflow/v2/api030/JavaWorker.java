package io.bioimage.modelrunner.tensorflow.v2.api030;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.apposed.appose.Service.RequestType;
import io.bioimage.modelrunner.apposed.appose.Service.ResponseType;

public class JavaWorker {
	
	private static LinkedHashMap<String, Object> tasks = new LinkedHashMap<String, Object>();
	
	private final String uuid;
	
	private final Tensorflow2Interface ti;
		
	private boolean cancelRequested = false;
	
	public static void main(String[] args) {
    	
    	try(Scanner scanner = new Scanner(System.in)){
    		Tensorflow2Interface ti;
    		try {
				ti = new Tensorflow2Interface(false);
			} catch (IOException | URISyntaxException e) {
				return;
			}
            
            while (true) {
            	String line;
                try {
                    if (!scanner.hasNextLine()) break;
                    line = scanner.nextLine().trim();
                } catch (Exception e) {
                    break;
                }
                
                if (line.isEmpty()) break;
                Map<String, Object> request = Types.decode(line);
                String uuid = (String) request.get("task");
                String requestType = (String) request.get("requestType");
                
                if (requestType.equals(RequestType.EXECUTE.toString())) {
                	String script = (String) request.get("script");
                	LinkedHashMap<String, Object> inputs = (LinkedHashMap<String, Object>) request.get("inputs");
                	JavaWorker task = new JavaWorker(uuid, ti);
                	tasks.put(uuid, task);
                	task.start(script, inputs);
                } else if (requestType.equals(RequestType.CANCEL.toString())) {
                	JavaWorker task = (JavaWorker) tasks.get(uuid);
                	if (task == null) {
                		System.err.println("No such task: " + uuid);
                		continue;
                	}
                	task.cancelRequested = true;
                } else {
                	break;
                }
            }
    	}
		
	}
	
	private JavaWorker(String uuid, Tensorflow2Interface ti) {
		this.uuid = uuid;
		this.ti = ti;
	}
	
	private void executeScript(String script, LinkedHashMap<String, Object> inputs) {
		LinkedHashMap<String, Object> binding = new LinkedHashMap<String, Object>();
		binding.put("task", this);
		if (inputs != null)
			binding.putAll(binding);
		
		this.reportLaunch();
		try {
			if (script.equals("loadModel")) {
				ti.loadModel((String) inputs.get("modelFolder"), null);
			} else if (script.equals("inference")) {
				ti.runFromShmas((LinkedHashMap<String, Object>) inputs.get("inputs"), (LinkedHashMap<String, Object>) inputs.get("outputs"));
			} else if (script.equals("close")) {
				ti.closeModel();
			}
		} catch(Exception ex) {
			this.fail(Types.stackTrace(ex.getCause()));
			return;
		}
		this.reportCompletion();
	}
	
	private void start(String script, LinkedHashMap<String, Object> inputs) {
		new Thread(() -> executeScript(script, inputs), "Appose-" + this.uuid).start();
	}
	
	private void reportLaunch() {
		respond(ResponseType.LAUNCH, null);
	}
	
	private void reportCompletion() {
		respond(ResponseType.COMPLETION, null);
	}
	
	private void update(String message, Integer current, Integer maximum) {
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		
		if (message != null)
			args.put("message", message);
		
		if (current != null)
			args.put("current", current);
		
		if (maximum != null)
			args.put("maximum", maximum);
		this.respond(ResponseType.UPDATE, args);
	}
	
	private void respond(ResponseType responseType, LinkedHashMap<String, Object> args) {
		LinkedHashMap<String, Object> response = new LinkedHashMap<String, Object>();
		response.put("task", uuid);
		response.put("responseType", responseType);
		if (args != null)
			response.putAll(response);
		try {
			System.out.println(Types.encode(response));
			System.out.flush();
		} catch(Exception ex) {
			this.fail(Types.stackTrace(ex.getCause()));
		}
	}
	
	private void cancel() {
		this.respond(ResponseType.CANCELATION, null);
	}
	
	private void fail(String error) {
		LinkedHashMap<String, Object> args = null;
		if (error != null) {
			args = new LinkedHashMap<String, Object>();
			args.put("error", error);
		}
        respond(ResponseType.FAILURE, args);
	}

}
