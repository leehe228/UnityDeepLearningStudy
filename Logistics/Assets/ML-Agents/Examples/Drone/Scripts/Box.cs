using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using PA_DronePack;

public class Box : MonoBehaviour 
{
    public GameObject agent;
    public string agentName;
    private bool isEnter;

    public GameObject map;

    void Start()
    {
        isEnter = false;
    }

    void Update()
    {
        if (isEnter) {
            Vector3 agentPos = agent.transform.position;
            agentPos.y = Mathf.Max(0.3f, agentPos.y - 1.2f);
            gameObject.transform.position = agentPos;
            agent.GetComponent<Drone>().boxPos = gameObject.transform.position;
        }
    }

    void OnCollisionEnter(Collision other) {
        if (!isEnter) {
            if (other.gameObject.CompareTag("agent")) {
                agentName = other.gameObject.name;
                agent = GameObject.Find(agentName);
                if (agent.GetComponent<Drone>().catchNum == 0) {
                    isEnter = true;
                    agent.GetComponent<Drone>().boxPos = gameObject.transform.position;
                    agent.GetComponent<Drone>().catchNum += 1;
                    agent.GetComponent<Drone>().boxType = 1;
                    agent.GetComponent<DroneAgent>().GiveReward(1.0f);
                }
            }
        }

        if (other.gameObject.CompareTag("dest1")) {
            agent.GetComponent<Drone>().catchNum -= 1;
            agent.GetComponent<Drone>().boxType = 0;
            isEnter = false;
            GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
            GetComponent<Rigidbody>().angularVelocity = new Vector3(0f, 0f, 0f);
            agent.GetComponent<DroneAgent>().GiveReward(5.0f);
            map.GetComponent<MapController>().generate1();
        }

        else if (other.gameObject.CompareTag("dest2")) {
            agent.GetComponent<DroneAgent>().GiveReward(-0.1f);
        }
    }
}
