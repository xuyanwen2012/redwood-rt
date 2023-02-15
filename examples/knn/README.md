# Barnes Hut Demo

## General Structure



## Some notes

```

    int cur_stream = 0;
    while (!q_data.empty()) {
      for (auto it = exe[cur_stream].begin(); it != exe[cur_stream].end();
           ++it) {
        if (it->Finished()) {
          if (q_data.empty()) {
            break;
          }

          const auto q = q_data.front();
          q_data.pop();

          it->StartQuery(q);
        } else {
          it->Resume();
        }
      }

      exe[cur_stream][0].k_set_->DebugPrint();
      std::cout << std::endl;
    }

```
