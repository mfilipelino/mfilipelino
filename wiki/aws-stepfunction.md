# AWS Stepfunction

Getting started with AWS step functions

{% embed url="https://aws.amazon.com/step-functions/getting-started/#Tutorials" %}

How do step functions work?&#x20;

{% embed url="https://docs.aws.amazon.com/step-functions/latest/dg/how-step-functions-works.html" %}

{% embed url="https://www.youtube.com/watch?v=1RJtCKpdELQ&list=PLJo-rJlep0EBq0-P-2wq5tzTXjL_jmynX" %}



Developer guide to AWS step functions:



#### What is an orchestrator?&#x20;



**Nutshell**

In a nutshell, an orchestrator in software engineering is a tool or system that automates the management, coordination, and scheduling of tasks and services in a distributed environment. It handles things like resource allocation, scaling, failure recovery, and workflow execution, simplifying the complexity of running large-scale applications.

#### Functions of an Orchestrator

1. **Task Scheduling**: Determines when and where to run tasks based on predefined logic or real-time metrics.
2. **Resource Allocation**: Manages computing resources, ensuring that tasks have the CPU, memory, and storage they need.
3. **Service Coordination**: Manages inter-service communication, ensuring that services interact in a predefined manner, maintaining dependencies and order.
4. **Scaling**: Automatically scales services up or down based on metrics like traffic, CPU usage, etc.
5. **Failure Recovery**: Detects failed tasks or services and restarts them either on the same node or a different node.
6. **Load Balancing**: Distributes tasks or network traffic across a number of servers or nodes.
7. **Monitoring and Logging**: Provides a centralized system for tracking the state and health of all services and tasks.
8. **Configuration Management**: Manages and updates the configuration settings of services, often without stopping or restarting them.

#### Popular Orchestrators:

* **Kubernetes**: Highly popular for container orchestration, particularly Docker containers.
* **Apache Mesos**: General-purpose distributed systems kernel that's highly scalable.
* **Apache Airflow**: Often used for orchestrating complex data pipelines.
* **AWS Step Functions**: Used for serverless orchestration of AWS services.

So, if you are dealing with complex systems involving multiple services, databases, or containers, an orchestrator can help manage this complexity efficiently.



**Orchestration Patterns:**

AWS Step Functions can indeed handle a variety of orchestration patterns but with some limitations. Here's a breakdown:

1. **Sequential Orchestration**: Easily achieved. You can define steps to execute in sequence.
2. **Parallel Orchestration**: Supported. You can use the `Parallel` state in Step Functions to execute multiple branches concurrently.
3. **Conditional Orchestration**: Supported through `Choice` states, which route the execution to different states based on conditions.
4. **Looping Orchestration**: Partially supported. You can implement loops using `Choice` and `Fail` states, but it's not as straightforward as in some other platforms.
5. **Saga Pattern**: Can be implemented manually using compensating transactions. AWS Step Functions provides built-in support for try-catch-finally constructs which can be useful here.
6. **State Machine Pattern**: Step Functions itself is designed as a state machine, so this pattern is natively supported.
7. **Fan-Out/Fan-In**: Achievable using the `Parallel` state to fan-out and then using another state to aggregate or fan-in.
8. **Choreography**: Not natively supported, as Step Functions is more focused on centralized orchestration. However, you can integrate with other AWS services like SNS or Lambda to achieve a choreography pattern.
9. **Master-Worker Pattern**: While not a native feature, this can be orchestrated by combining Step Functions with other AWS services like Lambda and SQS.
10. **Data-Driven Orchestration**: You can use `Choice` states to route based on data conditions, and Lambda functions to manipulate or analyze data within the workflow.

#### AWS Step Functions:

1. **Pass State**: Passes its input to its output, without performing work. Useful for injecting fixed values or restructuring the workflow's data.
2. **Task State**: Represents a single unit of work performed by a state machine. It often invokes AWS Lambda functions, but can also call other supported AWS services.
3. **Choice State**: Adds branching logic to a state machine based on the value of a field in its JSON input.
4. **Wait State**: Introduces a delay for a certain time or until a specified timestamp.
5. **Succeed State**: Stops an execution successfully, terminating the state machine.
6. **Fail State**: Stops an execution with a failure, terminating the state machine.
7. **Parallel State**: Provides concurrency by branching into multiple sub-flows, then merges their results to continue execution.
8. **Map State**: Processes an array of items by iterating through each one and executing a set of states for each.
9. **Catch and Finally**: Not states per se, but these are clauses you can attach to states to handle error conditions (Catch) or perform cleanup tasks (Finally).

Input and output processing



{% embed url="https://static.us-east-1.prod.workshops.aws/public/74efd33c-69f6-43c7-a67b-fafd049835c7/assets/img/en/intro/input-output.png" %}
[https://docs.aws.amazon.com/step-functions/latest/dg/concepts-input-output-filtering.html](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-input-output-filtering.html)
{% endembed %}

#### Service Integration Patterns

<figure><img src="https://static.us-east-1.prod.workshops.aws/public/74efd33c-69f6-43c7-a67b-fafd049835c7/assets/img/en/intro/service-integration-patterns.png" alt="" width="563"><figcaption><p><a href="https://docs.aws.amazon.com/step-functions/latest/dg/connect-to-resource.html">Service Integration Patterns Documentation </a></p></figcaption></figure>

