conv_int2pred = function(model_type, pd) {
    if (model_type == 2) {
        if (pd == 1) {
            return "positive";
        } else {
            return "negative";
        }
    } else {
        if (pd == 2) {
            return "positive";
        } else if (pd == 1) {
            return "neutral";
        } else {
            return "negative";
        }
    }
}

construct_row = function(idx, text, pred) {
    var pre_row = "<tr><th>";
    var mid_row = "</th><th>";
    var end_row = "</th></tr>";
    return pre_row + idx + mid_row + text + mid_row + pred + end_row;
}

parse_one_tweet_pred = function(model_type, data, fn) {
    var pred = conv_int2pred(model_type, data['pred']);
    var model = data['model'];
    fn(pred, model);
}

parse_list_tweet_pred = function(model_type, data) {
    for (var i = 0; i < data.length; i++) {
        var text = data['text'];
        var pred_lst = data['predict'];
        for (var j = 0; j < pred_lst.length; j++) {
            parse_one_tweet_pred(model_type, pred_lst[j], function(pred, model) {
                var row = construct_row(i + 1, text, pred);
                $('#tweet-res-table > tbody:last-child').append(row);
            });
        }
    }
}

$("#send-points").click(function() {
    var txt = JSON.stringify([$("#tweet-input").val()]);
    console.log(txt);
    $.post('/predone', {'tweet':txt, 'model_type': 2}).done(function(data) {
        data = JSON.parse(data);
        $("tbody").html("");
        for (var i = 0; i < data.length; i++) {
            parse_one_tweet_pred(2, data[i], function(pred, model) {
                var row = construct_row(i + 1, model, pred);
                $('#tweet-res-table > tbody:last-child').append(row);
            });
        }
    });
});

$("#send-3points").click(function() {
    var txt = JSON.stringify([$("#tweet-input").val()]);
    console.log(txt);
    $.post('/predone', {'tweet':txt, 'model_type': 3}).done(function(data) {
        data = JSON.parse(data);
        $("tbody").html("");
        for (var i = 0; i < data.length; i++) {
            parse_one_tweet_pred(3, data[i], function(pred, model) {
                var row = construct_row(i + 1, model, pred);
                $('#tweet-res-table > tbody:last-child').append(row);
            });
        }
    });
});

$("#send-search-req2").click(function() {
    var qry = $("#search-query").val();
    $.post('/predlist', {'query': qry, 'model_type': 2, 'count': 15}).done(function(data) {
        data = JSON.parse(data);
        console.log(data)
        $("tbody").html("");
        var n_pos = 0;
        var n_neg = 0;
        for (var i = 0; i < data.length; i++) {
            var text = data[i]['text'];
            var pred_lst = data[i]['predict'];
            var confident = 0;
            for (var j = 0; j < pred_lst.length; j++) {
                parse_one_tweet_pred(2, pred_lst[j], function(pred, model) {
                    console.log(pred);
                    if (pred == "positive") {
                        n_pos += 1;
                        confident += 1;
                    } else {
                        n_neg += 1;
                    }
                    console.log(model, ": ", pred);
                });
            }
            var row = construct_row(i + 1, text, confident / pred_lst.length);
            $('#tweet-res-table > tbody:last-child').append(row);
        }
        var ctx = $("#myChart");
        var data = {
            labels: [
                "Negative",
                "Positive"
            ],
            datasets: [
                {
                    data: [n_neg, n_pos],
                    backgroundColor: [
                        "#FF6384",
                        "#36A2EB",
                    ],
                    hoverBackgroundColor: [
                        "#FF6384",
                        "#36A2EB",
                    ]
                }]
        };
        var myPieChart = new Chart(ctx,{
            type: 'pie',
            data: data,
            animation:{
                animateScale:true
            }
        });
    });
})

$("#send-search-req3").click(function() {
    var qry = $("#search-query").val();
    $.post('/predlist', {'query': qry, 'model_type': 3, 'count': 15}).done(function(data) {
        data = JSON.parse(data);
        console.log(data)
        $("tbody").html("");
        var n_pos = 0;
        var n_neg = 0;
        var n_neu = 0;
        for (var i = 0; i < data.length; i++) {
            var text = data[i]['text'];
            var pred_lst = data[i]['predict'];
            var confident = 0;
            for (var j = 0; j < pred_lst.length; j++) {
                parse_one_tweet_pred(3, pred_lst[j], function(pred, model) {
                    if (pred == "positive") {
                        n_pos += 1;
                        confident += 2;
                    } else if (pred == "negative") {
                        n_neg += 1;
                    } else {
                        n_neu += 1;
                        confident += 1;
                    }
                    console.log(model, ": ", pred);
                });
            }
            var row = construct_row(i + 1, text, confident / pred_lst.length);
            $('#tweet-res-table > tbody:last-child').append(row);
        }
        var ctx = $("#myChart");
        var data = {
            labels: [
                "Negative",
                "Positive",
                "Neutral"
            ],
            datasets: [
                {
                    data: [n_neg, n_pos, n_neu],
                    backgroundColor: [
                        "#FF6384",
                        "#36A2EB",
                        "#FFCE56"
                    ],
                    hoverBackgroundColor: [
                        "#FF6384",
                        "#36A2EB",
                        "#FFCE56"
                    ]
                }]
        };
        var myPieChart = new Chart(ctx,{
            type: 'pie',
            data: data,
            animation:{
                animateScale:true
            }
        });
    });
})
