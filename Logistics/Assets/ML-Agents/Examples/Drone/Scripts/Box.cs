using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using PA_DronePack;

public class Box : MonoBehaviour
{
    public GameObject s;
    public GameObject agent;
    public string agentName;
    private bool isEnter;

    public GameObject map;

    // Start is called before the first frame update
    void Start()
    {
        isEnter = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (isEnter) {
            Vector3 agentPos = agent.transform.position;
            agentPos.y = Mathf.Max(0.3f, agentPos.y - 1.2f);
            //agentPos.y -= 1.2f;
            s.transform.position = agentPos;
            agent.GetComponent<Drone>().boxPos = s.transform.position;
            //
            //float d = Vector3.Distance(map.GetComponent<MapController>().dest1.transform.position, s.transform.position);
            
        }

        /*if (s.transform.position.y < -5f) {
            s.transform.position = new Vector3(s.transform.position.x, 3f, s.transform.position.z);
        }*/
    }

    void OnCollisionEnter(Collision other) {
        if (!isEnter) {
            if (other.gameObject.CompareTag("agent")) {
                agentName = other.gameObject.name;
                agent = GameObject.Find(agentName);
                if (agent.GetComponent<Drone>().catchNum == 0) {
                    isEnter = true;
                    agent.GetComponent<Drone>().boxPos = s.transform.position;
                    agent.GetComponent<Drone>().catchNum += 1;
                    agent.GetComponent<DroneAgent>().GiveReward(0.3f);
                }
            }
        }

        if (other.gameObject.CompareTag("dest1")) {
            agent.GetComponent<Drone>().catchNum -= 1;
            isEnter = false;
            GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
            GetComponent<Rigidbody>().angularVelocity = new Vector3(0f, 0f, 0f);
            agent.GetComponent<DroneAgent>().GiveReward(1f);
            map.GetComponent<MapController>().generate1();
        }

        else if (other.gameObject.CompareTag("dest2")) {
            agent.GetComponent<DroneAgent>().GiveReward(-0.5f);
        }
    }
}
