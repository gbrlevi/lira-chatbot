# Devolução: política aplicada a um pedido real

Caso não trivial: a Lira pede os dados de verificação, consulta o pedido, aplica a política de arrependimento à previsão de entrega e abre um chamado.

_Conversa real, capturada com o relógio fixado em 2026-03-25T10:00:00._

**Cliente:** Me arrependi da minha compra, posso devolver meu pedido?

> Ferramentas: `consultar_politicas`

**Lira:** Sua compra foi feita pela internet? Se sim, você pode solicitar a devolução em até 7 dias corridos após o recebimento, desde que o produto esteja na embalagem original, sem sinais de uso e com todos os acessórios e manuais. Quer que eu registre uma solicitação para você?

**Cliente:** Comprei pelo site sim. É o pedido 7, telefone (67) 99789-0123.

> Ferramentas: `consultar_pedidos`

**Lira:** Letícia, vi aqui que a previsão de entrega foi em 17/02/2026. Como o sistema não registra a data exata em que você recebeu o produto, o prazo de 7 dias para o arrependimento provavelmente já venceu. Se quiser, posso registrar um chamado para a nossa equipe avaliar o seu caso, tá bom?

**Cliente:** Entendi. Mesmo assim queria que alguém da loja avaliasse meu caso, pode ser?

> Ferramentas: `registrar_chamado`

**Lira:** Pronto, Letícia! Registrei seu caso para nossa equipe avaliar. O protocolo é EM-20260325-0001 e o retorno será dado em até 24 horas úteis pelo telefone que você informou. Posso ajudar com mais alguma coisa?
