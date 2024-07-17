package org.apposed.appose;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.RequestType;
import io.bioimage.modelrunner.apposed.appose.Service.ResponseType;

public class JavaWorker {
	
	private static HashMap<Object, Object> tasks = new HashMap<Object, Object>();
	
	private final String uuid;
	
	private HashMap<Object, Object> outputs = new HashMap<Object, Object>();
	
	private boolean cancelRequested = false;
	
	public static void main(String[] args) {
    	
    	try(Scanner scanner = new Scanner(System.in)){
            
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
                	Map<String, Object> inputs = (Map<String, Object>) request.get("inputs");
                	JavaWorker task = new JavaWorker(uuid);
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
	
	private JavaWorker(String uuid) {
		this.uuid = uuid;
	}
	
	private void executeScript(String script, Map<String, Object> inputs) {
		Map<String, Object> binding = new HashMap<String, Object>();
		binding.put("task", this);
		if (inputs != null)
			binding.putAll(binding);
		
		this.reportLaunch();
		try {
			// TODO find a way to launch the code in the separate process
		} catch(Exception ex) {
			this.fail(Types.stackTrace(ex.getCause()));
			return;
		}
		this.reportCompletion();
	}
	
	private void start(String script, Map<String, Object> inputs) {
		new Thread(() -> executeScript(script, inputs), "Appose-" + this.uuid).start();
	}
	
	private void reportLaunch() {
		respond(ResponseType.LAUNCH, null);
	}
	
	private void reportCompletion() {
		respond(ResponseType.COMPLETION, null);
	}
	
	private void update(String message, Integer current, Integer maximum) {
		Map<String, Object> args = new HashMap<String, Object>();
		
		if (message != null)
			args.put("message", message);
		
		if (current != null)
			args.put("current", current);
		
		if (maximum != null)
			args.put("maximum", maximum);
		this.respond(ResponseType.UPDATE, args);
	}
	
	private void respond(ResponseType responseType, Map<String, Object> args) {
		Map<String, Object> response = new HashMap<String, Object>();
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
		Map<String, Object> args = null;
		if (error != null) {
			args = new HashMap<String, Object>();
			args.put("error", error);
		}
        respond(ResponseType.FAILURE, args);
	}

}
