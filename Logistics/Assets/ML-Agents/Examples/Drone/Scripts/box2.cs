using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using PA_DronePack;

public class box2 : MonoBehaviour
{
    public GameObject s;
    public GameObject agent1, agent2;
    public string agent1Name, agent2Name;
    private bool isEnter1, isEnter2;

    public GameObject map;

    // Start is called before the first frame update
    void Start()
    {
        isEnter1 = false;
        isEnter2 = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (isEnter1 && isEnter2) {
            Vector3 agent1pos = agent1.transform.position;
            Vector3 agent2pos = agent2.transform.position;

            if (Vector3.Distance(agent1pos, agent2pos) > 3.5f) {
                isEnter1 = false;
                isEnter2 = false;
                agent1.GetComponent<Drone>().catchNum -= 1;
                agent2.GetComponent<Drone>().catchNum -= 1;
                agent1.GetComponent<DroneAgent>().GiveReward(-0.3f);
                agent2.GetComponent<DroneAgent>().GiveReward(-0.3f);
                GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
                GetComponent<Rigidbody>().angularVelocity = new Vector3(0f, 0f, 0f);
            }
            else {
                Vector3 p = (agent1pos + agent2pos) / 2;
                p.y = Mathf.Max(0.3f, p.y - 1.2f);
                s.transform.position = p;
                agent1.GetComponent<Drone>().boxPos = p;
                agent2.GetComponent<Drone>().boxPos = p;
                //
            }
        }

        if (isEnter1 && isEnter2 == false) {
            Vector3 agent1pos = agent1.transform.position;
            if (Vector3.Distance(agent1pos, s.transform.position) > 2.5f) {
                isEnter1 = false;
                agent1.GetComponent<Drone>().catchNum -= 1;
                agent1.GetComponent<DroneAgent>().GiveReward(-0.1f);
            } 
        }

        /*if (s.transform.position.y < -5f) {
            s.transform.position = new Vector3(s.transform.position.x, 3f, s.transform.position.z);
        }*/
    }

    void OnCollisionEnter(Collision other) {
        if (!isEnter1) {
            if (other.gameObject.CompareTag("agent")) {
                agent1Name = other.gameObject.name;
                agent1 = GameObject.Find(agent1Name);
                if (agent1.GetComponent<Drone>().catchNum == 0) {
                    isEnter1 = true;
                    agent1.GetComponent<Drone>().boxPos = s.transform.position;
                    agent1.GetComponent<Drone>().catchNum += 1;
                    agent1.GetComponent<DroneAgent>().GiveReward(0.2f);
                }
            }
        }
        if (isEnter1 == true && isEnter2 == false) {
            if (other.gameObject.CompareTag("agent")) {
                agent2Name = other.gameObject.name;
                agent2 = GameObject.Find(agent2Name);
                if (agent2.GetComponent<Drone>().catchNum == 0) {
                    isEnter2 = true;
                    agent2.GetComponent<Drone>().catchNum += 1;
                    agent1.GetComponent<DroneAgent>().GiveReward(0.1f);
                    agent2.GetComponent<DroneAgent>().GiveReward(0.3f);
                }
            }
        }

        if (isEnter1 && isEnter2) {
            if (other.gameObject.CompareTag("dest2")) {
                agent1.GetComponent<Drone>().catchNum -= 1;
                agent2.GetComponent<Drone>().catchNum -= 1;
                isEnter1 = false;
                isEnter2 = false;
                GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
                GetComponent<Rigidbody>().angularVelocity = new Vector3(0f, 0f, 0f);
                agent1.GetComponent<DroneAgent>().GiveReward(2f);
                agent2.GetComponent<DroneAgent>().GiveReward(2f);
                map.GetComponent<MapController>().generate2();
            }

            else if (other.gameObject.CompareTag("dest1")) {
                agent1.GetComponent<DroneAgent>().GiveReward(-0.5f);
                agent2.GetComponent<DroneAgent>().GiveReward(-0.5f);
            }
        }
    }

    void OnCollisionExit(Collision other) {
        
    }
}
