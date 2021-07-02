using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Box : MonoBehaviour
{
    public GameObject s;
    public GameObject agent;
    private bool isEnter;

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
            agentPos.y -= 1.2f;
            s.transform.position = agentPos;
        }
    }

    void OnCollisionEnter(Collision other) {
        if (other.gameObject.CompareTag("agent")) {
            isEnter = true;
        }

        if (other.gameObject.CompareTag("destination")) {
            Destroy(s);
        }
    }
}
