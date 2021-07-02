using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MapController : MonoBehaviour
{

    public GameObject hub;
    public GameObject box1, box2, box3;
    public GameObject destination;

    

    // Start is called before the first frame update
    void Start()
    {
        box1 = GameObject.FindWithTag("box1");
        box2 = GameObject.FindWithTag("box2");
        box3 = GameObject.FindWithTag("box3");

        Invoke("generate", 1);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void generate() {
        Vector3 pos1 = new Vector3(Random.Range(-3f, 3f), Random.Range(5f, 6f), Random.Range(-3f, 3f));
        Vector3 pos2 = new Vector3(Random.Range(-3f, 3f), Random.Range(5f, 6f), Random.Range(-3f, 3f));
        Vector3 pos3 = new Vector3(Random.Range(-3f, 3f), Random.Range(5f, 6f), Random.Range(-3f, 3f));

        Instantiate(box1, pos1, Quaternion.identity);
        Instantiate(box2, pos2, Quaternion.identity);
        Instantiate(box3, pos3, Quaternion.identity);

        Invoke("generate", 3);
    }
}
